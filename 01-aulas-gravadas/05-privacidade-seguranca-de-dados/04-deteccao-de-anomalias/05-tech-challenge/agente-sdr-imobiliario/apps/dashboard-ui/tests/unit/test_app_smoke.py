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
        kpis, error = dashboard_app.fetch_kpis("http://api.local/", "tok", http=http)
        assert error is None
        assert kpis["leads_today"] == 3
        assert http.calls[0]["url"] == "http://api.local/api/kpis"

    def test_401_maps_to_session_expired(self):
        kpis, error = dashboard_app.fetch_kpis("http://api.local", "tok", http=FakeHttp(FakeResponse(401)))
        assert kpis is None
        assert error["status"] == 401
        assert "login" in error["message"].lower()

    def test_500_maps_to_error_toast(self):
        _, error = dashboard_app.fetch_kpis("http://api.local", "tok", http=FakeHttp(FakeResponse(500)))
        assert error["status"] == 500

    def test_network_error_is_graceful(self):
        _, error = dashboard_app.fetch_kpis("http://api.local", "tok", http=FakeHttp(error=OSError("down")))
        assert error["status"] is None
        assert "indisponível" in error["message"]

    def test_invalid_json_is_graceful(self):
        _, error = dashboard_app.fetch_kpis(
            "http://api.local", "tok", http=FakeHttp(FakeResponse(200, broken_json=True))
        )
        assert error["status"] == 200

    def test_missing_api_url_is_graceful(self):
        _, error = dashboard_app.fetch_kpis("", "tok", http=FakeHttp())
        assert error["status"] is None

    def test_token_flows_as_bearer(self):
        http = FakeHttp(FakeResponse(200, KPIS))
        dashboard_app.fetch_kpis("http://api.local", "jwt-poc", http=http)
        assert http.calls[0]["headers"]["Authorization"] == "Bearer jwt-poc"

    def test_missing_token_sends_no_auth_header(self):
        http = FakeHttp(FakeResponse(200, KPIS))
        dashboard_app.fetch_kpis("http://api.local", None, http=http)
        assert http.calls[0]["headers"] == {}


class FakeCognitoClient:
    def __init__(self, initiate_auth_result=None, challenge_result=None, error=None):
        self.initiate_auth_result = initiate_auth_result
        self.challenge_result = challenge_result
        self.error = error
        self.calls: list[dict] = []

    def initiate_auth(self, **kwargs):
        self.calls.append({"op": "initiate_auth", **kwargs})
        if self.error:
            raise self.error
        return self.initiate_auth_result

    def respond_to_auth_challenge(self, **kwargs):
        self.calls.append({"op": "respond_to_auth_challenge", **kwargs})
        if self.error:
            raise self.error
        return self.challenge_result


CONFIG = {"pool_id": "us-east-1_abc", "client_id": "client123", "region": "us-east-1"}


class TestCognitoConfig:
    def test_missing_env_returns_none(self, monkeypatch):
        monkeypatch.delenv("COGNITO_USER_POOL_ID", raising=False)
        monkeypatch.delenv("COGNITO_CLIENT_ID", raising=False)
        monkeypatch.delenv("AWS_REGION", raising=False)
        assert dashboard_app.cognito_config() is None

    def test_full_env_returns_config(self, monkeypatch):
        monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_abc")
        monkeypatch.setenv("COGNITO_CLIENT_ID", "client123")
        monkeypatch.setenv("AWS_REGION", "us-east-1")
        assert dashboard_app.cognito_config() == CONFIG


class TestCognitoLogin:
    def test_success_returns_tokens(self):
        client = FakeCognitoClient(
            initiate_auth_result={"AuthenticationResult": {"IdToken": "id.tok"}}
        )
        result, error = dashboard_app.cognito_login("user", "pwd", CONFIG, client=client)
        assert error is None
        assert result == {"tokens": {"IdToken": "id.tok"}}
        assert client.calls[0]["AuthParameters"] == {"USERNAME": "user", "PASSWORD": "pwd"}

    def test_new_password_required_returns_challenge(self):
        client = FakeCognitoClient(
            initiate_auth_result={"ChallengeName": "NEW_PASSWORD_REQUIRED", "Session": "sess1"}
        )
        result, error = dashboard_app.cognito_login("user", "temp", CONFIG, client=client)
        assert error is None
        assert result == {"challenge": "NEW_PASSWORD_REQUIRED", "session": "sess1"}

    def test_auth_failure_returns_error(self):
        client = FakeCognitoClient(error=Exception("NotAuthorizedException"))
        result, error = dashboard_app.cognito_login("user", "wrong", CONFIG, client=client)
        assert result is None
        assert "Login falhou" in error


class TestCognitoRespondNewPassword:
    def test_success_returns_tokens(self):
        client = FakeCognitoClient(
            challenge_result={"AuthenticationResult": {"IdToken": "id.tok2"}}
        )
        result, error = dashboard_app.cognito_respond_new_password(
            "user", "NovaSenha1", "sess1", CONFIG, client=client
        )
        assert error is None
        assert result == {"tokens": {"IdToken": "id.tok2"}}
        assert client.calls[0]["Session"] == "sess1"

    def test_failure_returns_error(self):
        client = FakeCognitoClient(error=Exception("InvalidPasswordException"))
        result, error = dashboard_app.cognito_respond_new_password(
            "user", "weak", "sess1", CONFIG, client=client
        )
        assert result is None
        assert "Troca de senha falhou" in error


class TestRenderSmoke:
    def test_render_runs_with_streamlit(self):
        pytest.importorskip("streamlit")
        dashboard_app.render(KPIS, None, "http://api.local")

    def test_render_error_path_runs(self):
        pytest.importorskip("streamlit")
        dashboard_app.render(None, {"status": 401, "message": "Não autenticado"}, "http://api.local")
