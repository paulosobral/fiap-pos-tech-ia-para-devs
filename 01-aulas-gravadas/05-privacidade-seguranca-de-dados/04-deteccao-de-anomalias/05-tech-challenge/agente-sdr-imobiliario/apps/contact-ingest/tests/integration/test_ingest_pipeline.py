import base64
import logging
import sys
from email.message import EmailMessage
from unittest.mock import MagicMock

import pytest

from handler import handler
from infra.session_store import channel_user_id
from service.contact_ingest import ContactIngest, PendingRetryError
from service.email_parser import HeuristicEmailParser
from service.router_gateway import RouterError


PORTAL_BODY = (
    "Novo contato recebido pelo portal.\n"
    "Nome: Ana Souza\n"
    "E-mail: ana.souza@empresa.com.br\n"
    "Telefone: (11) 98888-7777\n"
    "Interesse: laje corporativa de 300 m² na Faria Lima\n"
)


def raw_content(body: str) -> str:
    message = EmailMessage()
    message["Subject"] = "Contato via portal"
    message["From"] = "Portal Imob <no-reply@portalimob.com.br>"
    message["To"] = "contato@wlevitt.app"
    message.set_content(body)
    return base64.b64encode(message.as_bytes()).decode()


def ses_record(
    message_id="mid-1",
    source="no-reply@portalimob.com.br",
    subject="Novo lead: Ana Souza",
    content=None,
    mail_overrides=None,
):
    mail = {
        "messageId": message_id,
        "source": source,
        "commonHeaders": {
            "from": [f"Portal Imob <{source}>"] if source else [],
            "to": ["contato@wlevitt.app"],
            "subject": subject,
            "date": "Sat, 20 Sep 2026 12:00:00 +0000",
        },
    }
    if content is not None:
        mail["content"] = content
    if mail_overrides:
        mail.update(mail_overrides)
    return {
        "eventSource": "aws:ses",
        "eventVersion": "1.0",
        "ses": {
            "mail": mail,
            "receipt": {"spamVerdict": {"status": "PASS"}, "virusVerdict": {"status": "PASS"}},
        },
    }


def ses_event(*records):
    return {"Records": list(records)}


class MemoryDedupe:
    def __init__(self, delete_fails=False):
        self._seen = {}
        self._delete_fails = delete_fails

    def put_first(self, message_id, source=None):
        if message_id in self._seen:
            return False
        self._seen[message_id] = {"status": "INGESTED", "source": source}
        return True

    def delete(self, message_id):
        if self._delete_fails:
            return False
        self._seen.pop(message_id, None)
        return True

    def mark_quarantine(self, message_id, reason):
        self._seen[message_id] = {"status": "QUARANTINE", "reason": reason}
        return True

    def take_quarantined(self, message_id):
        item = self._seen.get(message_id)
        if item is not None and item.get("status") == "QUARANTINE":
            self._seen.pop(message_id, None)
            return True
        return False


class MemorySessions:
    def __init__(self):
        self.sessions = []

    def open_session(self, contact):
        lead_id = f"lead-{len(self.sessions) + 1}"
        session = {
            "lead_id": lead_id,
            "session_id": f"s-{len(self.sessions) + 1}",
            "telegram_user_id": channel_user_id(contact["email"], lead_id),
        }
        self.sessions.append((contact, session))
        return session


class FakeRouter:
    def __init__(self, fail_first=False):
        self.calls = []
        self.fail_first = fail_first

    def reinject(self, telegram_user_id, session_id, text):
        payload = {"telegram_user_id": telegram_user_id, "session_id": session_id, "text": text}
        if self.fail_first and len(self.calls) == 0:
            self.calls.append(payload)
            raise RouterError("router down")
        self.calls.append(payload)


def make_pipeline(router=None):
    dedupe = MemoryDedupe()
    sessions = MemorySessions()
    router = router or FakeRouter()
    ingest = ContactIngest(
        parser=HeuristicEmailParser(), dedupe=dedupe, sessions=sessions, router=router
    )
    return ingest, sessions, router


class TestIngestPipeline:
    def test_portal_email_opens_session_and_reinjects_first_message(self):
        ingest, sessions, router = make_pipeline()
        summary = ingest.handle_event(ses_event(ses_record(content=raw_content(PORTAL_BODY))))
        assert summary["ok"] == 1
        (contact, session), = sessions.sessions
        assert contact["name"] == "Ana Souza"
        assert contact["email"] == "ana.souza@empresa.com.br"
        assert contact["phone"] == "(11) 98888-7777"
        assert router.calls[0]["session_id"] == session["session_id"]
        assert router.calls[0]["telegram_user_id"] == session["telegram_user_id"]
        assert "Faria Lima" in router.calls[0]["text"]

    def test_duplicate_message_only_one_session(self):
        ingest, sessions, router = make_pipeline()
        record = ses_record(message_id="m1", content=raw_content(PORTAL_BODY))
        summary = ingest.handle_event(ses_event(record, dict(record)))
        assert summary["ok"] == 1
        assert summary["drop"] == 1
        assert len(sessions.sessions) == 1
        assert len(router.calls) == 1

    def test_two_portal_formats_ingest_ok(self):
        ingest, sessions, router = make_pipeline()
        labeled = ses_record(message_id="m1", content=raw_content(PORTAL_BODY))
        subject_only = ses_record(
            message_id="m2",
            source="outroportal@portais.com",
            subject="Contato de Marcos Souza",
        )
        summary = ingest.handle_event(ses_event(labeled, subject_only))
        assert summary["ok"] == 2
        assert len(sessions.sessions) == 2
        assert sessions.sessions[0][1]["session_id"] != sessions.sessions[1][1]["session_id"]
        assert sessions.sessions[1][0]["email"] == "outroportal@portais.com"
        assert len(router.calls) == 2

    def test_unparseable_email_dropped(self):
        ingest, sessions, router = make_pipeline()
        record = ses_record(source="", mail_overrides={"commonHeaders": {"subject": "Ola"}})
        summary = ingest.handle_event(ses_event(record))
        assert summary["drop"] == 1
        assert sessions.sessions == []
        assert router.calls == []

    def test_transient_router_failure_retries_and_recovers(self):
        ingest, sessions, router = make_pipeline(router=FakeRouter(fail_first=True))
        record = ses_record(message_id="m1", content=raw_content(PORTAL_BODY))
        summary = ingest.handle_event(ses_event(record, dict(record)))
        assert summary["retry"] == 1
        assert summary["ok"] == 1
        assert len(router.calls) == 2
        assert router.calls[1]["text"] == router.calls[0]["text"]
        # Reprocesso abre nova sessão (a 1ª, sem mensagem entregue, fica órfã — POC)
        assert len(sessions.sessions) == 2

    def test_no_pii_in_structured_logs(self, caplog):
        ingest, *_ = make_pipeline()
        with caplog.at_level(logging.INFO):
            ingest.handle_event(ses_event(ses_record(content=raw_content(PORTAL_BODY))))
        assert "ana.souza@empresa.com.br" not in caplog.text
        assert "Ana Souza" not in caplog.text
        assert "98888-7777" not in caplog.text
        assert '"event": "contact_ingested"' in caplog.text

    def test_rollback_failure_quarantine_releases_on_reprocess(self):
        dedupe = MemoryDedupe(delete_fails=True)
        sessions = MemorySessions()
        router = FakeRouter(fail_first=True)
        ingest = ContactIngest(
            parser=HeuristicEmailParser(), dedupe=dedupe, sessions=sessions, router=router
        )
        record = ses_record(message_id="m1", content=raw_content(PORTAL_BODY))
        summary = ingest.handle_event(ses_event(record, dict(record)))
        assert summary["retry"] == 1
        assert summary["ok"] == 1
        # 1ª tentativa: rollback falhou → marca QUARANTINE; reprocesso libera e ingeriu
        assert dedupe._seen["m1"]["status"] == "INGESTED"
        assert len(router.calls) == 2
        assert len(sessions.sessions) == 2


class TestHandlerWiring:
    def _patch_aws(self, monkeypatch, post_side_effect=None, post_status=200):
        requests_mod = MagicMock()
        session = requests_mod.Session.return_value
        if post_side_effect is not None:
            session.post.side_effect = post_side_effect
        else:
            session.post.return_value = MagicMock(status_code=post_status)
        monkeypatch.setitem(sys.modules, "requests", requests_mod)
        monkeypatch.setitem(sys.modules, "boto3", MagicMock())
        monkeypatch.setenv("ROUTER_BASE_URL", "http://router.local")
        monkeypatch.setenv("INTERNAL_SECRET_TOKEN", "sec")
        return requests_mod

    def test_handler_processes_portal_email_end_to_end(self, monkeypatch):
        requests_mod = self._patch_aws(monkeypatch)
        summary = handler(ses_event(ses_record(content=raw_content(PORTAL_BODY))))
        assert summary["ok"] == 1
        _, call_kwargs = requests_mod.Session.return_value.post.call_args
        assert call_kwargs["headers"]["X-Internal-Secret"] == "sec"
        assert call_kwargs["json"]["session_id"]

    def test_handler_raises_pending_retry_on_transient_failure(self, monkeypatch):
        self._patch_aws(monkeypatch, post_side_effect=RuntimeError("router down"))
        with pytest.raises(PendingRetryError):
            handler(ses_event(ses_record(content=raw_content(PORTAL_BODY))))

    def test_handler_drops_unparseable_without_raising(self, monkeypatch):
        self._patch_aws(monkeypatch)
        record = ses_record(source="", mail_overrides={"commonHeaders": {"subject": "Ola"}})
        summary = handler(ses_event(record))
        assert summary["drop"] == 1

    def test_handler_missing_internal_secret_fails_fast(self, monkeypatch):
        self._patch_aws(monkeypatch)
        monkeypatch.delenv("INTERNAL_SECRET_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="INTERNAL_SECRET_TOKEN"):
            handler(ses_event(ses_record(content=raw_content(PORTAL_BODY))))
