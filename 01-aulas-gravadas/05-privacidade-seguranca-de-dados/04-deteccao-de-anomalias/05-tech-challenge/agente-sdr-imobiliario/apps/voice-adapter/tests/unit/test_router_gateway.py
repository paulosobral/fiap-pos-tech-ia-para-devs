from unittest.mock import MagicMock

import pytest

from service.router_gateway import HttpRouterGateway, RouterError


def make_gateway():
    http = MagicMock()
    gateway = HttpRouterGateway(http, base_url="http://router.local", secret_token="sec")
    return gateway, http


def make_response(status_code=200, response_text="ok"):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"ok": True, "response": response_text}
    return resp


class TestHttpRouterGateway:
    def test_reinject_posts_payload_and_secret(self):
        gateway, http = make_gateway()
        http.post.return_value = make_response()
        gateway.reinject(42, "s1", "texto transcrito")
        url, kwargs = http.post.call_args[0][0], http.post.call_args.kwargs
        assert url == "http://router.local/internal/inbound-text"
        assert kwargs["json"] == {
            "telegram_user_id": 42,
            "session_id": "s1",
            "text": "texto transcrito",
        }
        assert kwargs["headers"]["X-Internal-Secret"] == "sec"

    def test_reinject_http_error_raises(self):
        gateway, http = make_gateway()
        http.post.return_value = make_response(status_code=500)
        with pytest.raises(RouterError):
            gateway.reinject(42, "s1", "texto")

    def test_reinject_network_error_raises(self):
        gateway, http = make_gateway()
        http.post.side_effect = OSError("connection refused")
        with pytest.raises(RouterError):
            gateway.reinject(42, "s1", "texto")

    def test_missing_secret_fails_fast_before_any_request(self):
        http = MagicMock()
        with pytest.raises(RouterError):
            HttpRouterGateway(http, base_url="http://router.local")
        http.post.assert_not_called()

    def test_reinject_sends_u1_accepted_contract_shape(self):
        gateway, http = make_gateway()
        http.post.return_value = make_response()
        gateway.reinject(42, "s1", "texto transcrito")
        url, kwargs = http.post.call_args[0][0], http.post.call_args.kwargs
        assert url == "http://router.local/internal/inbound-text"
        assert kwargs["json"] == {
            "telegram_user_id": 42,
            "session_id": "s1",
            "text": "texto transcrito",
        }
        assert kwargs["headers"] == {
            "Content-Type": "application/json",
            "X-Internal-Secret": "sec",
        }

    def test_trailing_slash_normalized(self):
        http = MagicMock()
        http.post.return_value = make_response()
        gateway = HttpRouterGateway(http, base_url="http://router.local/", secret_token="sec")
        gateway.reinject(1, "s", "t")
        assert http.post.call_args[0][0] == "http://router.local/internal/inbound-text"
