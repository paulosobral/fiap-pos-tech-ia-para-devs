from unittest.mock import MagicMock

import pytest

from service.flow_gateway import FlowError, HttpFlowGateway


def make_gateway(status_code=200, secret="sec", post_ok=True):
    http = MagicMock()
    if post_ok:
        http.post.return_value = MagicMock(status_code=status_code)
    return HttpFlowGateway(http, base_url="http://flow.local/", secret_token=secret), http


class TestHttpFlowGateway:
    def test_notify_posts_status_payload(self):
        gateway, _ = make_gateway()
        gateway.notify_status("lead-1", "s1", "qualificado")
        args, call_kwargs = gateway._http.post.call_args
        assert args[0] == "http://flow.local/internal/crm-status"
        assert call_kwargs["json"] == {
            "lead_id": "lead-1",
            "session_id": "s1",
            "stage": "qualificado",
        }
        assert call_kwargs["headers"]["X-Internal-Secret"] == "sec"

    def test_notify_without_secret_omits_header(self):
        gateway, _ = make_gateway(secret=None)
        gateway.notify_status("lead-1", "s1", "novo")
        _, call_kwargs = gateway._http.post.call_args
        assert "X-Internal-Secret" not in call_kwargs["headers"]

    def test_connection_error_raises_flow_error(self):
        gateway, http = make_gateway(post_ok=False)
        http.post.side_effect = RuntimeError("connection refused")
        with pytest.raises(FlowError):
            gateway.notify_status("lead-1", "s1", "novo")

    def test_http_rejection_raises_flow_error(self):
        gateway, _ = make_gateway(status_code=500)
        with pytest.raises(FlowError):
            gateway.notify_status("lead-1", "s1", "novo")

    def test_redirect_status_is_rejected(self):
        gateway, _ = make_gateway(status_code=302)
        with pytest.raises(FlowError):
            gateway.notify_status("lead-1", "s1", "novo")

    def test_trailing_slash_is_stripped(self):
        _, http = make_gateway()
        gateway = HttpFlowGateway(http, base_url="http://flow.local")
        gateway.notify_status("lead-1", "s1", "novo")
        assert gateway._http.post.call_args[0][0].startswith("http://flow.local/")