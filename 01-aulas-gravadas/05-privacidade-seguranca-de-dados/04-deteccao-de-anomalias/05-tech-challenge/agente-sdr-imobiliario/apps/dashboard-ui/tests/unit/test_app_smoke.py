import json

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


def _jwt(claims: dict) -> str:
    import base64
    import json

    body = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"h.{body}.s"


class TestSessionVault:
    def test_create_get_drop(self):
        vault = dashboard_app.SessionVault()
        sid = vault.create("refresh-1")
        assert vault.get(sid) == "refresh-1"
        vault.drop(sid)
        assert vault.get(sid) is None

    def test_unknown_or_empty_sid(self):
        vault = dashboard_app.SessionVault()
        assert vault.get(None) is None
        assert vault.get("nope") is None

    def test_expires_after_ttl(self):
        clock = [1000.0]
        vault = dashboard_app.SessionVault(ttl_seconds=60, now_fn=lambda: clock[0])
        sid = vault.create("refresh-1")
        clock[0] += 59
        assert vault.get(sid) == "refresh-1"
        clock[0] += 2
        assert vault.get(sid) is None

    def test_sids_are_unique(self):
        vault = dashboard_app.SessionVault()
        assert vault.create("a") != vault.create("a")


class TestCognitoRefresh:
    CONFIG = {"region": "us-east-1", "client_id": "cid"}

    def test_success_returns_id_token(self):
        class Client:
            def initiate_auth(self, **kwargs):
                assert kwargs["AuthFlow"] == "REFRESH_TOKEN_AUTH"
                assert kwargs["AuthParameters"] == {"REFRESH_TOKEN": "r"}
                return {"AuthenticationResult": {"IdToken": "new-id"}}

        assert dashboard_app.cognito_refresh("r", self.CONFIG, client=Client()) == ("new-id", None)

    def test_failure_returns_error(self):
        class Client:
            def initiate_auth(self, **kwargs):
                raise RuntimeError("revoked")

        token, error = dashboard_app.cognito_refresh("r", self.CONFIG, client=Client())
        assert token is None and "revoked" in error


class TestIdentityAndCookie:
    def test_user_label_prefers_email(self):
        assert dashboard_app.user_label(_jwt({"email": "a@b.com", "cognito:username": "u"})) == "a@b.com"

    def test_user_label_falls_back(self):
        assert dashboard_app.user_label(_jwt({"cognito:username": "u"})) == "u"
        assert dashboard_app.user_label("lixo") == "usuário"
        assert dashboard_app.user_label(None) == "usuário"

    def test_cookie_script_sets_and_clears(self):
        assert "sdr_session=abc_-1" in dashboard_app.cookie_script("abc_-1", 60)
        assert "max-age=0" in dashboard_app.cookie_script("x", 0)

    def test_cookie_script_rejects_unsafe_value(self):
        with pytest.raises(AssertionError):
            dashboard_app.cookie_script("a';alert(1)//", 60)


class TestMainLogin:
    def test_main_without_session_shows_login_without_error(self, monkeypatch):
        st = pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        script = (
            "import app\n"
            "app.main()\n"
        )
        monkeypatch.setenv("DASHBOARD_API_URL", "")
        monkeypatch.setenv("COGNITO_CLIENT_ID", "cid")
        monkeypatch.setenv("COGNITO_REGION", "us-east-1")
        at = AppTest.from_string(script).run(timeout=20)
        assert not at.exception


class TestLeads:
    LEADS = [{"lead_id": "L1", "name": "Ana", "phone": "11999990000", "email": "a@x.com", "score": 80}]

    def test_fetch_leads_ok_and_errors(self):
        ok = FakeHttp(FakeResponse(200, {"leads": self.LEADS}))
        assert dashboard_app.fetch_leads("http://api", "t", http=ok) == (self.LEADS, None)
        assert ok.calls[0]["url"] == "http://api/api/leads"
        leads, error = dashboard_app.fetch_leads("http://api", "t", http=FakeHttp(FakeResponse(401)))
        assert leads is None and error["status"] == 401
        leads, error = dashboard_app.fetch_leads("http://api", "t", http=FakeHttp(error=RuntimeError("x")))
        assert leads is None and "indisponível" in error["message"]

    def test_send_lead_to_crm(self):
        class Http:
            def __init__(self, status):
                self.status, self.calls = status, []

            def post(self, url, timeout=None, headers=None):
                self.calls.append(url)
                return FakeResponse(self.status)

        http = Http(202)
        assert dashboard_app.send_lead_to_crm("http://api", "t", "L1", http=http)[0] is True
        assert http.calls == ["http://api/api/leads/L1/crm"]
        assert dashboard_app.send_lead_to_crm("http://api", "t", "L1", http=Http(404)) == (False, "Lead não encontrado.")
        assert dashboard_app.send_lead_to_crm("http://api", "t", "L1", http=Http(500))[0] is False

    def test_render_leads_runs(self):
        pytest.importorskip("streamlit")
        dashboard_app.render_leads(self.LEADS, None, "http://api", "t")
        dashboard_app.render_leads([], None, "http://api", "t")
        dashboard_app.render_leads(None, {"status": 500, "message": "x"}, "http://api", "t")


class TestBranding:
    ROOT = __import__("pathlib").Path(dashboard_app.__file__).parent

    def test_logo_is_the_original_designer_png(self):
        import hashlib
        import os

        assert os.path.exists(dashboard_app.LOGO_PATH)
        original = self.ROOT.parents[1] / "Designer.png"  # logo oficial na raiz do repo
        if original.exists():  # no container só existe a cópia; no repo as duas têm que ser idênticas
            digest = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
            assert digest(dashboard_app.LOGO_PATH) == digest(original)

    def test_theme_uses_logo_palette(self):
        theme = (self.ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8").lower()
        assert dashboard_app.BRAND_GOLD.lower() in theme
        assert 'backgroundcolor = "#0b1d3a"' in theme

    def test_dockerfile_ships_assets_and_theme(self):
        docker = (self.ROOT / "Dockerfile").read_text(encoding="utf-8")
        assert "COPY assets assets" in docker and "COPY .streamlit .streamlit" in docker

    def test_header_renders_with_logo(self):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        script = (
            "import app, streamlit as st\n"
            "st.session_state['id_token'] = 'h.e30.s'\n"
            "app._render_header(st)\n"
        )
        at = AppTest.from_string(script).run(timeout=20)
        assert not at.exception
        assert any(b.label == "Sair" for b in at.button)


class TestFavicon:
    def test_page_icon_is_the_logo_thumbnail(self):
        icon = dashboard_app.page_icon()
        assert not isinstance(icon, str)
        assert max(icon.size) <= 128

    def test_page_icon_falls_back_to_emoji_without_logo(self, monkeypatch):
        monkeypatch.setattr(dashboard_app, "LOGO_PATH", "/nao/existe.png")
        assert dashboard_app.page_icon() == "🏢"

    def test_both_page_configs_use_the_icon(self):
        import inspect

        source = inspect.getsource(dashboard_app)
        assert source.count("page_icon=page_icon()") == 2
        assert 'page_title="Dashboard SDR' not in source


class TestRenderWithoutData:
    """Infra recém-criada: nenhum lead, nenhuma intenção, nenhuma roleta (bug real no login)."""

    def _run(self, kpis):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(__import__('pathlib').Path(dashboard_app.__file__).parent)!r})\n"
            "import app\n"
            f"app.render(json.loads({json.dumps(kpis)!r}), None, 'http://api.local')\n"
        )
        return AppTest.from_string(script).run(timeout=30)

    def test_empty_intents_and_roulette_do_not_crash(self):
        at = self._run({**KPIS, "intents": {}, "route_distribution": {}, "alerts": [], "funnel": {}})
        assert not at.exception, at.exception
        assert any("Sem dados" in c.value for c in at.caption)

    def test_missing_chart_keys_do_not_crash(self):
        kpis = {k: v for k, v in KPIS.items() if k not in ("intents", "route_distribution")}
        assert not self._run(kpis).exception

    def test_all_zero_fresh_environment(self):
        fresh = {"leads_today": 0, "leads_week": 0, "response_time_p90": 0.0, "qualification_rate": 0.0,
                 "scheduled_visits": 0, "anomalies_count": 0, "cost_monthly": 0.0, "generated_at": "2026-10-09T00:00:00+00:00",
                 "intents": {}, "route_distribution": {}, "funnel": {}, "alerts": []}
        assert not self._run(fresh).exception

    def test_populated_charts_still_render(self):
        at = self._run({**KPIS, "intents": {"compra": 2, "locação": 5}, "route_distribution": {"Ana": 3}})
        assert not at.exception, at.exception


class TestPortugueseLabels:
    """O banco e a API guardam códigos em inglês (handoff, rent, high...); a tela mostra português."""

    def test_every_pipeline_state_has_a_portuguese_label(self):
        for state in dashboard_app.KANBAN_STATES:
            assert state in dashboard_app.STATE_LABELS, state
            assert dashboard_app.STATE_LABELS[state] != state

    def test_handoff_reads_as_forwarded_to_the_broker(self):
        assert dashboard_app.pt(dashboard_app.STATE_LABELS, "handoff") == "Encaminhado ao corretor"

    def test_unknown_or_empty_values_are_never_hidden_or_broken(self):
        assert dashboard_app.pt(dashboard_app.STATE_LABELS, "novo_estado") == "novo_estado"
        assert dashboard_app.pt(dashboard_app.STATE_LABELS, None) is None
        assert dashboard_app.pt(dashboard_app.STATE_LABELS, "") == ""

    def test_intent_chart_merges_codes_and_translated_names(self):
        merged = dashboard_app.translate_counts(dashboard_app.INTENT_LABELS, {"purchase": 2, "Compra": 1, "rent": 3})
        assert merged == {"Compra": 3, "Locação": 3}
        assert dashboard_app.translate_counts(dashboard_app.INTENT_LABELS, None) == {}

    def test_alert_rows_are_translated_and_readable(self):
        rows = dashboard_app.alert_rows([
            {"anomaly_id": "a1", "lead_id": "L1", "type": "atypical_hours", "confidence": 0.7,
             "detected_at": "2026-10-09T01:00:00+00:00", "status": "open", "action_taken": "alert_issued"},
            {"anomaly_id": "a2", "lead_id": "L2", "type": "tipo_novo", "confidence": None, "status": "resolved",
             "action_taken": "schedule_restricted"},
        ])
        assert rows[0]["Tipo"] == "Fora do horário comercial"
        assert rows[0]["Confiança"] == "70%"
        assert rows[0]["Situação"] == "Aberto" and rows[0]["Ação tomada"] == "Alerta emitido"
        assert rows[1]["Tipo"] == "tipo_novo" and rows[1]["Situação"] == "Resolvido"
        assert rows[1]["Ação tomada"] == "Agendamento bloqueado"
        assert "open" not in str(rows) and "alert_issued" not in str(rows)

    def _run(self, body: str):
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        script = (
            "import sys, json\n"
            f"sys.path.insert(0, {str(__import__('pathlib').Path(dashboard_app.__file__).parent)!r})\n"
            "import app\n" + body
        )
        return AppTest.from_string(script).run(timeout=30)

    def test_rendered_leads_table_shows_portuguese_states(self):
        leads = [{"lead_id": "L1", "name": "Ana", "phone": "11999990000", "email": None, "score": 80,
                  "urgency": "high", "intent": "rent", "budget": None, "area": None, "region": None,
                  "state": "handoff", "updated_at": "2026-10-09"}]
        at = self._run(f"app.render_leads(json.loads({json.dumps(leads)!r}), None, 'http://api', 't')\n")
        assert not at.exception, at.exception
        frame = at.dataframe[0].value
        assert list(frame["Estado"]) == ["Encaminhado ao corretor"]
        assert list(frame["Intenção"]) == ["Locação"] and list(frame["Urgência"]) == ["Alta"]

    def test_rendered_dashboard_shows_portuguese_states_and_alerts(self):
        kpis = {**KPIS, "funnel": {"handoff": 4, "greeting": 1}, "intents": {"purchase": 2},
                "alerts": [{"anomaly_id": "a1", "lead_id": "L1", "type": "high_volume", "confidence": 0.8,
                            "status": "open", "action_taken": "alert_issued", "detected_at": "2026-10-09"}]}
        at = self._run(f"app.render(json.loads({json.dumps(kpis)!r}), None, 'http://api')\n")
        assert not at.exception, at.exception
        labels = [m.label for m in at.metric]
        assert "Encaminhado ao corretor" in labels and "Saudação" in labels
        assert "handoff" not in labels and "greeting" not in labels
        alerts = at.dataframe[0].value
        assert list(alerts["Tipo"]) == ["Volume alto de mensagens"] and list(alerts["Situação"]) == ["Aberto"]
