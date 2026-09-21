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
        dedupe.mark_quarantine.assert_not_called()
        router.reinject.assert_not_called()

    def test_router_failure_retries_and_rolls_back_dedupe(self):
        ingest, _, dedupe, sessions, router = make_ingest()
        router.reinject.side_effect = RouterError("router down")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        dedupe.delete.assert_called_once_with("mid-1")
        dedupe.mark_quarantine.assert_not_called()

    def test_rollback_failure_still_retries_and_quarantines(self):
        ingest, _, dedupe, sessions, _ = make_ingest()
        sessions.open_session.side_effect = SessionError("dynamo down")
        dedupe.delete.side_effect = RuntimeError("delete boom")
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        assert dedupe.mark_quarantine.call_args.args[0] == "mid-1"
        assert dedupe.mark_quarantine.call_args.args[1].startswith("rollback_failed:")

    def test_quarantined_stale_mark_releases_and_reingests(self):
        ingest, _, dedupe, sessions, router = make_ingest()
        dedupe.put_first.side_effect = [False, True]
        dedupe.take_quarantined.return_value = True
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["ok"] == 1
        dedupe.take_quarantined.assert_called_once_with("mid-1")
        sessions.open_session.assert_called_once()
        router.reinject.assert_called_once()

    def test_completed_ingest_mark_is_never_released(self):
        ingest, _, dedupe, sessions, router = make_ingest(dedupe_first=False)
        dedupe.take_quarantined.return_value = False
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["drop"] == 1
        dedupe.take_quarantined.assert_called_once_with("mid-1")
        sessions.open_session.assert_not_called()
        router.reinject.assert_not_called()

    def test_quarantine_marking_failure_is_logged_not_silent(self, caplog):
        ingest, _, dedupe, sessions, _ = make_ingest()
        sessions.open_session.side_effect = SessionError("dynamo down")
        dedupe.delete.side_effect = RuntimeError("delete boom")
        dedupe.mark_quarantine.side_effect = RuntimeError("dynamo down")
        with caplog.at_level(logging.INFO):
            summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        assert '"event": "dedupe_quarantine_failed"' in caplog.text

    def test_missing_message_id_uses_content_hash_key(self):
        ingest, _, dedupe, *_ = make_ingest()
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail(message_id="")}}]})
        assert summary["ok"] == 1
        key = dedupe.put_first.call_args.args[0]
        assert key.startswith("sha256:")
        assert len(key) > len("sha256:")

    def test_two_emails_without_message_id_do_not_collapse(self):
        ingest, parser, dedupe, sessions, _ = make_ingest()
        parser.parse.side_effect = [
            parsed(),
            parsed(message_text="Outro interesse, outra laje", subject="Outro assunto"),
        ]
        records = [
            {"ses": {"mail": ses_mail(message_id="")}},
            {"ses": {"mail": ses_mail(message_id="")}},
        ]
        summary = ingest.handle_event({"Records": records})
        assert summary["ok"] == 2
        keys = [call.args[0] for call in dedupe.put_first.call_args_list]
        assert keys[0] != keys[1]
        assert all(key.startswith("sha256:") for key in keys)

    def test_same_content_without_message_id_dedupes_to_one_ingest(self):
        ingest, parser, dedupe, sessions, router = make_ingest()
        parser.parse.side_effect = [parsed(), parsed()]
        dedupe.put_first.side_effect = [True, False]
        dedupe.take_quarantined.return_value = False
        records = [
            {"ses": {"mail": ses_mail(message_id="")}},
            {"ses": {"mail": ses_mail(message_id="")}},
        ]
        summary = ingest.handle_event({"Records": records})
        assert summary["ok"] == 1
        assert summary["drop"] == 1
        assert sessions.open_session.call_count == 1
        router.reinject.assert_called_once()

    def test_source_without_domain_uses_none(self):
        ingest, _, dedupe, *_ = make_ingest()
        mail = ses_mail()
        mail["source"] = "plain-sender"
        summary = ingest.handle_event({"Records": [{"ses": {"mail": mail}}]})
        assert summary["ok"] == 1
        dedupe.put_first.assert_called_once_with("mid-1", None)

    def test_error_logs_do_not_leak_exception_payload(self, caplog):
        ingest, parser, *_ = make_ingest()
        parser.parse.side_effect = ValueError("payload lead ana@empresa.com quer visita")
        with caplog.at_level(logging.INFO):
            summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        assert "ana@empresa.com" not in caplog.text
        assert "ValueError" in caplog.text

    def test_router_error_log_carries_type_only(self, caplog):
        ingest, _, _, _, router = make_ingest()
        router.reinject.side_effect = RouterError("smtp relay hint: ana@empresa.com")
        with caplog.at_level(logging.INFO):
            ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert "ana@empresa.com" not in caplog.text
        assert '"error": "RouterError"' in caplog.text

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

    def test_quarantine_release_error_falls_back_to_duplicate(self, caplog):
        ingest, _, dedupe, sessions, _ = make_ingest()
        dedupe.put_first.return_value = False
        dedupe.take_quarantined.side_effect = RuntimeError("get boom")
        with caplog.at_level(logging.INFO):
            summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["drop"] == 1
        assert '"event": "quarantine_release_failed"' in caplog.text
        sessions.open_session.assert_not_called()

    def test_quarantine_release_put_race_falls_back_to_duplicate(self):
        ingest, _, dedupe, sessions, router = make_ingest()
        dedupe.put_first.side_effect = [False, DedupeError("dynamo down")]
        dedupe.take_quarantined.return_value = True
        summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["drop"] == 1
        sessions.open_session.assert_not_called()
        router.reinject.assert_not_called()

    def test_mark_quarantine_raise_is_logged_not_raised(self, caplog):
        ingest, _, dedupe, sessions, _ = make_ingest()
        sessions.open_session.side_effect = SessionError("dynamo down")
        dedupe.delete.side_effect = RuntimeError("delete boom")
        dedupe.mark_quarantine.side_effect = RuntimeError("dynamo down")
        with caplog.at_level(logging.INFO):
            summary = ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert summary["retry"] == 1
        assert "dynamo down" not in caplog.text

    def test_no_pii_in_structured_logs(self, caplog):
        ingest, *_ = make_ingest()
        with caplog.at_level(logging.INFO):
            ingest.handle_event({"Records": [{"ses": {"mail": ses_mail()}}]})
        assert "ana@empresa.com" not in caplog.text
        assert "Ana Souza" not in caplog.text
        assert "Faria Lima" not in caplog.text
        assert '"event": "contact_ingested"' in caplog.text
