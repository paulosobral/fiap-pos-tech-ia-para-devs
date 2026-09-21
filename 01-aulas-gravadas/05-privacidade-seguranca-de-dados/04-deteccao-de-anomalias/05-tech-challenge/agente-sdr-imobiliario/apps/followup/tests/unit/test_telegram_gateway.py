from types import SimpleNamespace

import pytest

from service.telegram_gateway import GatewayError, TelegramGateway


class FakeHttp:
    def __init__(self, ok=True, error=None):
        self.ok = ok
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        if self.error:
            raise self.error
        self.calls.append({"url": url, **kwargs})
        return SimpleNamespace(ok=self.ok, status_code=200 if self.ok else 400)


class TestTelegramGateway:
    def test_send_message_posts_to_bot_api_send_message(self):
        http = FakeHttp()
        TelegramGateway(http, bot_token="tok", base_url="https://api.telegram.org").send_message(42, "olá")
        call = http.calls[0]
        assert call["url"] == "https://api.telegram.org/bottok/sendMessage"
        assert call["json"] == {"chat_id": 42, "text": "olá"}

    def test_send_message_rejects_http_error_status(self):
        with pytest.raises(GatewayError, match="rejected with status 400"):
            TelegramGateway(FakeHttp(ok=False), bot_token="tok").send_message(42, "olá")

    def test_send_message_wraps_http_failure(self):
        with pytest.raises(GatewayError, match="sendMessage failed"):
            TelegramGateway(FakeHttp(error=ConnectionError("boom")), bot_token="tok").send_message(42, "olá")

    def test_gateway_error_never_carries_url_or_token_from_original_exception(self):
        leaked = ConnectionError(
            "HTTPSConnectionPool(host='api.telegram.org', port=443): "
            "url=https://api.telegram.org/botSECRETTOKEN123/sendMessage"
        )
        with pytest.raises(GatewayError) as excinfo:
            TelegramGateway(FakeHttp(error=leaked), bot_token="SECRETTOKEN123").send_message(42, "olá")
        message = str(excinfo.value)
        assert "SECRETTOKEN123" not in message
        assert "https://" not in message
        assert "ConnectionError" in message
        assert "sendMessage failed" in message

    def test_base_url_trailing_slash_is_normalized(self):
        http = FakeHttp()
        TelegramGateway(http, bot_token="tok", base_url="https://api.telegram.org/").send_message(1, "x")
        assert http.calls[0]["url"].startswith("https://api.telegram.org/bottok/sendMessage")

    def test_timeout_is_passed_to_http_client(self):
        http = FakeHttp()
        TelegramGateway(http, bot_token="tok", timeout=7).send_message(1, "x")
        assert http.calls[0]["timeout"] == 7
