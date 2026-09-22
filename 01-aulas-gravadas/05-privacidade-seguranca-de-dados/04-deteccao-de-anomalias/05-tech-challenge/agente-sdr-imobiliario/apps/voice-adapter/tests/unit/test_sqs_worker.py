from unittest.mock import MagicMock, patch

from service import sqs_worker


def test_env_required():
    with patch.dict("os.environ", {}, clear=True):
        try:
            sqs_worker._env("NOPE")
            assert False, "deveria lançar RuntimeError"
        except RuntimeError:
            pass


def test_env_default():
    with patch.dict("os.environ", {"VOICE_QUEUE_URL": "q"}, clear=True):
        assert sqs_worker._env("MISSING", "fallback") == "fallback"
        assert sqs_worker._env("VOICE_QUEUE_URL") == "q"


def test_session_lookup_found(monkeypatch):
    client = MagicMock()
    client.get_item.return_value = {"Item": {"session_id": {"S": "s1"}}}
    lookup = sqs_worker.SessionLookup(client, "sessions")
    assert lookup.get_session("s1", 42) == {"session_id": "s1", "telegram_user_id": 42}


def test_session_lookup_missing(monkeypatch):
    client = MagicMock()
    client.get_item.return_value = {"Item": None}
    lookup = sqs_worker.SessionLookup(client, "sessions")
    assert lookup.get_session("s1", 42) is None


def test_main_poll_success_and_drop_deletes(monkeypatch):
    sqs = MagicMock()
    sqs.receive_message.return_value = {
        "Messages": [
            {"MessageId": "m1", "ReceiptHandle": "r1", "Body": "{}"},
            {"MessageId": "m2", "ReceiptHandle": "r2", "Body": "{}"},
        ]
    }

    adapter = MagicMock()
    outcomes = {"m1": "ok", "m2": "drop"}

    def fake_process(record):
        return outcomes.get(record["messageId"], "retry")

    adapter._process_record.side_effect = fake_process

    with patch.dict("os.environ", {"VOICE_QUEUE_URL": "q"}, clear=False), \
         patch("boto3.client", return_value=sqs) as boto_patch, \
         patch.object(sqs_worker, "_build_adapter", return_value=adapter), \
         patch("time.sleep") as sleep:
        # primeiras 1 recepções com mensagens, depois vazio dentro do loop infinito
        def receive_then_empty(*a, **kw):
            if not hasattr(receive_then_empty, "called"):
                receive_then_empty.called = True
                return sqs.receive_message.return_value
            raise SystemExit  # encerra o loop infinito no 2º ciclo

        sqs.receive_message.side_effect = receive_then_empty
        try:
            sqs_worker.main()
        except SystemExit:
            pass

    boto_patch.assert_called_once_with("sqs", region_name="us-east-1")
    # ok + drop -> deleta; retry não deleta
    assert sqs.delete_message.call_count == 2
    handles = sorted(
        call.kwargs.get("ReceiptHandle") for call in sqs.delete_message.call_args_list
    )
    assert handles == ["r1", "r2"]


def test_main_poll_retry_keeps_message(monkeypatch):
    sqs = MagicMock()
    sqs.receive_message.return_value = {
        "Messages": [{"MessageId": "m1", "ReceiptHandle": "r1", "Body": "{}"}]
    }

    adapter = MagicMock()
    adapter._process_record.return_value = "retry"

    with patch.dict("os.environ", {"VOICE_QUEUE_URL": "q"}, clear=False), \
         patch("boto3.client", return_value=sqs), \
         patch.object(sqs_worker, "_build_adapter", return_value=adapter), \
         patch("time.sleep"):
        def receive_then_empty(*a, **kw):
            if not hasattr(receive_then_empty, "called"):
                receive_then_empty.called = True
                return sqs.receive_message.return_value
            raise SystemExit

        sqs.receive_message.side_effect = receive_then_empty
        try:
            sqs_worker.main()
        except SystemExit:
            pass

    sqs.delete_message.assert_not_called()


def test_build_adapter_wiring(monkeypatch):
    env = {
        "VOICE_QUEUE_URL": "q",
        "TELEGRAM_BOT_TOKEN": "t",
        "INTERNAL_SECRET_TOKEN": "s",
        "ROUTER_BASE_URL": "https://api",
        "SESSIONS_TABLE": "sdr-sessions",
        "AWS_REGION": "us-east-1",
    }
    with patch.dict("os.environ", env):
        with patch.object(sqs_worker, "SessionLookup") as lookup_cls, \
             patch.object(sqs_worker, "VoiceAdapter") as adapter_cls, \
             patch.object(sqs_worker, "HttpRouterGateway"), \
             patch.object(sqs_worker, "TelegramGateway"):
            sqs_worker._build_adapter()
            adapter_cls.assert_called_once()
            assert adapter_cls.call_args.kwargs["sessions"] == lookup_cls.return_value