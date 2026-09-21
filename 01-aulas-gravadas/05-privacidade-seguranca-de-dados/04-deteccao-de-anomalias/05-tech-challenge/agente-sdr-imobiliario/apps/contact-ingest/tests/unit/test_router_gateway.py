from unittest.mock import MagicMock

import pytest

from service.router_gateway import RouterError, HttpRouterGateway


def make_gateway(status_code=200, secret="sec", post_ok=True):
    http = MagicMock()
    if post_ok:
        http.post.return_value = MagicMock(status_code=status_code)
    return HttpRouterGateway(http, base_url="http://router.local/", secret_token=secret), http


class TestHttpRouterGateway:
    def test_reinject_posts_inbound_text_payload(self):
        gateway, _ = make_gateway()
        gateway.reinject(42, "s1", "Quero uma laje na Faria Lima")
        args, call_kwargs = gateway._http.post.call_args
        assert args[0] == "http://router.local/internal/inbound-text"
        assert call_kwargs["json"] == {
            "telegram_user_id": 42,
            "session_id": "s1",
            "text": "Quero uma laje na Faria Lima",
        }
        assert call_kwargs["headers"]["X-Internal-Secret"] == "sec"

    def test_reinject_without_secret_omits_header(self):
        gateway, _ = make_gateway(secret=None)
        gateway.reinject(42, "s1", "Ola")
        _, call_kwargs = gateway._http.post.call_args
        assert "X-Internal-Secret" not in call_kwargs["headers"]

    def test_connection_error_raises_router_error(self):
        gateway, http = make_gateway(post_ok=False)
        http.post.side_effect = RuntimeError("connection refused")
        with pytest.raises(RouterError):
            gateway.reinject(42, "s1", "Ola")

    def test_http_rejection_raises_router_error(self):
        gateway, _ = make_gateway(status_code=500)
        with pytest.raises(RouterError):
            gateway.reinject(42, "s1", "Ola")

    def test_redirect_status_is_rejected(self):
        gateway, _ = make_gateway(status_code=302)
        with pytest.raises(RouterError):
            gateway.reinject(42, "s1", "Ola")

    def test_trailing_slash_is_stripped(self):
        _, http = make_gateway()
        gateway = HttpRouterGateway(http, base_url="http://router.local")
        gateway.reinject(42, "s1", "Ola")
        assert gateway._http.post.call_args[0][0].startswith("http://router.local/")
