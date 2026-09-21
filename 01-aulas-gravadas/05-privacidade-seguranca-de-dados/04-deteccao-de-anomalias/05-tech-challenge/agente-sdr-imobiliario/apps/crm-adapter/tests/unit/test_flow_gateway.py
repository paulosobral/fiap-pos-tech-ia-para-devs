from unittest.mock import MagicMock

import pytest

from service.flow_gateway import FlowError, HttpFlowGateway


def make_gateway(status_code=200, secret="seg-interno", post_ok=True):
    http = MagicMock()
    if post_ok:
        http.post.return_value = MagicMock(status_code=status_code)
    return HttpFlowGateway(http, base_url="http://flow.local/", secret_token=secret), http


class TestHttpFlowGateway:
    def test_notify_posts_official_contract_shape(self):
        """Shape exato do receptor real da u1 (POST /internal/crm-status)."""
        gateway, _ = make_gateway()
        gateway.notify_status("lead-1", "s1", "qualificado")
        args, call_kwargs = gateway._http.post.call_args
        assert args[0] == "http://flow.local/internal/crm-status"
        assert call_kwargs["json"] == {
            "lead_id": "lead-1",
            "session_id": "s1",
            "stage": "qualificado",
        }
        headers = call_kwargs["headers"]
        assert headers["X-Internal-Secret"] == "seg-interno"
        assert headers["Content-Type"] == "application/json"

    def test_header_always_sent(self):
        """O header é obrigatório: o receptor responde 401/503 sem X-Internal-Secret."""
        gateway, _ = make_gateway()
        gateway.notify_status("lead-1", "s1", "novo")
        _, call_kwargs = gateway._http.post.call_args
        assert call_kwargs["headers"]["X-Internal-Secret"] == "seg-interno"

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

    def test_unauthorized_is_rejected(self):
        gateway, _ = make_gateway(status_code=401)
        with pytest.raises(FlowError):
            gateway.notify_status("lead-1", "s1", "novo")

    def test_lead_not_found_is_rejected(self):
        gateway, _ = make_gateway(status_code=404)
        with pytest.raises(FlowError):
            gateway.notify_status("lead-1", "s1", "novo")

    def test_trailing_slash_is_stripped(self):
        _, http = make_gateway()
        gateway = HttpFlowGateway(http, base_url="http://flow.local", secret_token="seg-interno")
        gateway.notify_status("lead-1", "s1", "novo")
        assert gateway._http.post.call_args[0][0].startswith("http://flow.local/")
