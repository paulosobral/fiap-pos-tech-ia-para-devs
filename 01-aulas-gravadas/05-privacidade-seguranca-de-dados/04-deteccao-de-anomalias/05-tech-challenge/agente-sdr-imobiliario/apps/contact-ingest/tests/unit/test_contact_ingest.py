import logging
from unittest.mock import MagicMock

import pytest

from infra.dedupe_store import DedupeError
from infra.session_store import SessionError
from service.contact_ingest import ContactIngest
from service.email_parser import ParsedContact
from service.router_gateway import RouterError


def parsed(**overrides):
    base = dict(
        name="Ana Souza",
        email="ana@empresa.com",
        phone="11988887777",
        message_text="Interesse em laje na Faria Lima",
        subject="Novo lead",
    )
    base.update(overrides)
    return ParsedContact(**base)


def make_ingest(parse_result=parsed(), dedupe_first=True):
    parser = MagicMock()
    parser.parse.return_value = parse_result
    dedupe = MagicMock()
    dedupe.put_first.return_value = dedupe_first
    sessions = MagicMock()
    sessions.open_session.return_value = {
        "lead_id": "lead-1",
        "session_id": "s-1",
        "telegram_user_id": 42,
    }
    router = MagicMock()
    ingest = ContactIngest(parser=parser, dedupe=dedupe, sessions=sessions, router=router)
    return ingest, parser, dedupe, sessions, router


def ses_mail(message_id="mid-1"):
    return {
        "messageId": message_id,
        "source": "no-reply@portal.com.br",
        "commonHeaders": {"subject": "Novo lead"},
    }


class TestContactIngest:
    def test_happy_path_opens_session_and_reinjects(self):
        ingest, parser, dedupe, sessions, router = make_ingest()
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["ok"] == 1
        assert summary["retry"] == 0
        assert summary["drop"] == 0
        dedupe.put_first.assert_called_once_with("mid-1", "portal.com.br")
        sessions.open_session.assert_called_once_with(
            {"name": "Ana Souza", "email": "ana@empresa.com", "phone": "11988887777"}
        )
        router.reinject.assert_called_once_with(
            telegram_user_id=42, session_id="s-1", text="Interesse em laje na Faria Lima"
        )

    def test_unparseable_email_drops_without_side_effects(self):
        ingest, parser, dedupe, sessions, router = make_ingest(parse_result=None)
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["drop"] == 1
        dedupe.put_first.assert_not_called()
        sessions.open_session.assert_not_called()
        router.reinject.assert_not_called()

    def test_missing_ses_mail_drops(self):
        ingest, *_ = make_ingest()
        summary = ingest.handle_event({"Records": [{"ses": {}}]})
        assert summary["drop"] == 1

    def test_duplicate_email_drops(self):
        ingest, _, dedupe, sessions, router = make_ingest(dedupe_first=False)
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["drop"] == 1
        sessions.open_session.assert_not_called()
        router.reinject.assert_not_called()

    def test_dedupe_failure_retries_without_rollback(self):
        ingest, _, dedupe, sessions, router = make_ingest()
        dedupe.put_first.side_effect = DedupeError("dynamo down")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        sessions.open_session.assert_not_called()
        dedupe.delete.assert_not_called()

    def test_session_failure_retries_and_rolls_back_dedupe(self):
        ingest, _, dedupe, sessions, router = make_ingest()
        sessions.open_session.side_effect = SessionError("dynamo down")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        dedupe.delete.assert_called_once_with("mid-1")
        router.reinject.assert_not_called()

    def test_router_failure_retries_and_rolls_back_dedupe(self):
        ingest, _, dedupe, sessions, router = make_ingest()
        router.reinject.side_effect = RouterError("router down")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        dedupe.delete.assert_called_once_with("mid-1")

    def test_rollback_failure_still_retries(self):
        ingest, _, dedupe, sessions, _ = make_ingest()
        sessions.open_session.side_effect = SessionError("dynamo down")
        dedupe.delete.side_effect = RuntimeError("delete boom")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1

    def test_unexpected_failure_retries_without_raising(self):
        ingest, parser, *_ = make_ingest()
        parser.parse.side_effect = ValueError("boom")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1

    def test_mixed_event_counts(self):
        ingest, parser, *_ = make_ingest()
        parser.parse.side_effect = [parsed(), None, parsed()]
        records = [
            {"ses": {"mail": ses_mail("m1")}},
            {"ses": {"mail": ses_mail("m2")}},
            {"ses": {"mail": ses_mail("m3")}},
        ]
        summary = ingest.handle_event({"Records": records})
        assert summary["ok"] == 2
        assert summary["drop"] == 1
        assert summary["results"] == [
            {"message_id": "m1", "outcome": "ok"},
            {"message_id": "m2", "outcome": "drop"},
            {"message_id": "m3", "outcome": "ok"},
        ]

    def test_no_pii_in_structured_logs(self, caplog):
        ingest, *_ = make_ingest()
        with caplog.at_level(logging.INFO):
            ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert "ana@empresa.com" not in caplog.text
        assert "Ana Souza" not in caplog.text
        assert "Faria Lima" not in caplog.text
        assert '"event": "contact_ingested"' in caplog.text
