import pytest

import app as dashboard_app


class FakeResponse:
    def __init__(self, status_code=200, payload=None, broken_json=False):
        self.status_code = status_code
        self._payload = payload or {}
        self._broken_json = broken_json

    def json(self):
        if self._broken_json:
            raise ValueError("no json")
        return self._payload


class FakeHttp:
    def __init__(self, response=None, error=None):
        self.response = response or FakeResponse()
        self.error = error
        self.calls: list[dict] = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append({"url": url, "timeout": timeout, "headers": headers})
        if self.error:
            raise self.error
        return self.response


KPIS = {
    "leads_today": 3,
    "leads_week": 9,
    "response_time_p90": 0.8,
    "qualification_rate": 0.5,
    "scheduled_visits": 2,
    "anomalies_count": 1,
    "cost_monthly": 15.0,
    "generated_at": "2026-09-20T12:00:00+00:00",
    "intents": {"compra": 2},
    "route_distribution": {"consultores": 1, "diretor": 1},
    "funnel": {"greeting": 1, "scheduling": 1},
    "alerts": [{"anomaly_id": "a1", "lead_id": "l1", "type": "volume_spike"}],
}


class TestFetchKpis:
    def test_happy_returns_kpis_and_none_error(self):
        http = FakeHttp(FakeResponse(200, KPIS))
        kpis, error = dashboard_app.fetch_kpis("http://api.local/", http=http)
        assert error is None
        assert kpis["leads_today"] == 3
        assert http.calls[0]["url"] == "http://api.local/api/kpis"

    def test_401_maps_to_cognito_login(self):
        kpis, error = dashboard_app.fetch_kpis("http://api.local", http=FakeHttp(FakeResponse(401)))
        assert kpis is None
        assert error["status"] == 401
        assert "Cognito" in error["message"]

    def test_500_maps_to_error_toast(self):
        _, error = dashboard_app.fetch_kpis("http://api.local", http=FakeHttp(FakeResponse(500)))
        assert error["status"] == 500

    def test_network_error_is_graceful(self):
        _, error = dashboard_app.fetch_kpis("http://api.local", http=FakeHttp(error=OSError("down")))
        assert error["status"] is None
        assert "indisponível" in error["message"]

    def test_invalid_json_is_graceful(self):
        _, error = dashboard_app.fetch_kpis(
            "http://api.local", http=FakeHttp(FakeResponse(200, broken_json=True))
        )
        assert error["status"] == 200

    def test_missing_api_url_is_graceful(self):
        _, error = dashboard_app.fetch_kpis("", http=FakeHttp())
        assert error["status"] is None

    def test_token_env_flows_as_bearer(self, monkeypatch):
        monkeypatch.setenv("DASHBOARD_API_TOKEN", "jwt-poc")
        http = FakeHttp(FakeResponse(200, KPIS))
        dashboard_app.fetch_kpis("http://api.local", http=http)
        assert http.calls[0]["headers"]["Authorization"] == "Bearer jwt-poc"

    def test_cognito_login_url_from_env(self, monkeypatch):
        monkeypatch.delenv("COGNITO_LOGIN_URL", raising=False)
        assert dashboard_app.cognito_login_url() is None
        monkeypatch.setenv("COGNITO_LOGIN_URL", "https://auth.exemplo.com/login")
        assert dashboard_app.cognito_login_url() == "https://auth.exemplo.com/login"


class TestRenderSmoke:
    def test_render_runs_with_streamlit(self):
        pytest.importorskip("streamlit")
        dashboard_app.render(KPIS, None, "http://api.local")

    def test_render_error_path_runs(self):
        pytest.importorskip("streamlit")
        dashboard_app.render(None, {"status": 401, "message": "Não autenticado"}, "http://api.local")
