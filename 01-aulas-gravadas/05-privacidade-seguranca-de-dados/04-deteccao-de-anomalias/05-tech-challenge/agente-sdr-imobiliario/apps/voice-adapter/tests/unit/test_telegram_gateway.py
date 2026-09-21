from unittest.mock import MagicMock

import pytest

from service.telegram_gateway import GatewayError, TelegramGateway


def make_gateway():
    http = MagicMock()
    gateway = TelegramGateway(http, bot_token="tok")
    return gateway, http


def response(ok=True, json_data=None, status_code=200, content=b""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.content = content
    resp.json.return_value = json_data or {}
    return resp


class TestTelegramGateway:
    def test_get_file_path_returns_path(self):
        gateway, http = make_gateway()
        http.get.return_value = response(json_data={"ok": True, "result": {"file_path": "voice/f1.ogg"}})
        assert gateway.get_file_path("f1") == "voice/f1.ogg"
        url = http.get.call_args[0][0]
        assert "/bottok/getFile" in url
        assert http.get.call_args.kwargs["params"] == {"file_id": "f1"}

    def test_get_file_path_rejected_raises(self):
        gateway, http = make_gateway()
        http.get.return_value = response(ok=False, status_code=400, json_data={"ok": False})
        with pytest.raises(GatewayError):
            gateway.get_file_path("f1")

    def test_get_file_path_missing_result_raises(self):
        gateway, http = make_gateway()
        http.get.return_value = response(json_data={"ok": True, "result": {}})
        with pytest.raises(GatewayError):
            gateway.get_file_path("f1")

    def test_get_file_path_network_error_raises(self):
        gateway, http = make_gateway()
        http.get.side_effect = OSError("connection refused")
        with pytest.raises(GatewayError):
            gateway.get_file_path("f1")

    def test_download_returns_bytes(self):
        gateway, http = make_gateway()
        http.get.return_value = response(content=b"ogg-bytes")
        assert gateway.download("voice/f1.ogg") == b"ogg-bytes"
        assert "/file/bottok/voice/f1.ogg" in http.get.call_args[0][0]

    def test_download_rejected_raises(self):
        gateway, http = make_gateway()
        http.get.return_value = response(ok=False, status_code=404)
        with pytest.raises(GatewayError):
            gateway.download("voice/f1.ogg")

    def test_download_network_error_raises(self):
        gateway, http = make_gateway()
        http.get.side_effect = OSError("connection reset")
        with pytest.raises(GatewayError):
            gateway.download("voice/f1.ogg")

    def test_send_message_posts_payload(self):
        gateway, http = make_gateway()
        http.post.return_value = response()
        gateway.send_message(42, "olá")
        assert http.post.call_args.kwargs["json"] == {"chat_id": 42, "text": "olá"}

    def test_send_message_swallows_http_error(self):
        gateway, http = make_gateway()
        http.post.return_value = response(ok=False, status_code=500)
        gateway.send_message(42, "olá")

    def test_send_message_swallows_exception(self):
        gateway, http = make_gateway()
        http.post.side_effect = RuntimeError("network down")
        gateway.send_message(42, "olá")
