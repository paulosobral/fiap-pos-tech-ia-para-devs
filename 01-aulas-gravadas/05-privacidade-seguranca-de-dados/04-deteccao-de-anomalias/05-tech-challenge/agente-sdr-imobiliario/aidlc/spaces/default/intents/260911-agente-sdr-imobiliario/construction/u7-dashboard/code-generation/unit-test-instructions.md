# Unit Test Instructions — u7-dashboard

## Framework e configuração

- pytest + pytest-cov (já presentes no `.venv` da raiz; nenhum extra instalado).
- Bootstrap de path: cada app tem `tests/conftest.py` com `sys.path.insert(0, <app dir>)` — imports top-level `handler`, `infra.*`, `service.*` (padrão flat u1–u6), pyproject da raiz intocado.
- `apps/dashboard-api/tests/` e `apps/dashboard-ui/tests/` NÃO têm `__init__.py`: a suíte desta unidade roda os dois diretórios num único comando e dois pacotes `tests` no mesmo pytest colidem (`ImportPathMismatchError`).
- `streamlit`, `requests`, `boto3`, `fastapi`, `httpx` e `flask` NÃO estão no `.venv`: os testes nunca os importam de verdade — os módulos fake entram por `sys.modules` (`boto3`) ou injeção de dependência (`requests`/DynamoDB/CloudWatch); os testes de render da UI usam `pytest.importorskip("streamlit")` e pulam aqui.

## Como rodar ESTA unidade (comando exato, da raiz do repo)

```bash
.venv/bin/python -m pytest apps/dashboard-api/tests apps/dashboard-ui/tests --cov=apps/dashboard-api --cov-report=term-missing --cov-fail-under=80
```

Comando com escopo restrito aos diretórios de teste desta unidade (Build and Test não deve rodar suítes de outras unidades).

## Cobertura esperada

- Piso: **80% de linhas sobre `apps/dashboard-api`** (`--cov=apps/dashboard-api`); a UI é intencionalmente fora do piso (apenas smoke).
- Resultado obtido nesta passagem: **68 passed, 2 skipped** (smoke de render com `streamlit` ausente), **TOTAL 98.40%** — `handler.py` 100%, `metrics.py` 100%, `kpi_calc.py` 98%, `kpis.py` 98%, `conversation_store.py` 94%, `alert_store.py` 91%.
- Suítes anteriores re-verificadas após a unidade: u1 = 50 passed, u2 = 58 passed, u3 = 86 passed, u4 = 50 passed, u5 = 51 passed + 1 skipped, u6 = 77 passed.

## Mocking/stubbing

- Fakes in-memory por componente (`FakeScanClient`, `FailingClient`, `FakeCloudWatch`, `FakeAwsClient` com filtro `begins_with(SK, :prefix)` honrado, `FakeBoto3Module` em `sys.modules`, `FakeConversations`/`FakeAlerts`/`FakeMeter`, `FakeHttp`/`FakeResponse` na UI), injetados por construtor — zero rede/zero AWS.
- Relógio injetável (`now_fn`) fixado em `2026-09-20T12:00:00+00:00` — janelas de "hoje"/7 dias/24h determinísticas.
- Logs estruturados verificados com `caplog` (eventos `kpi_aggregation_*`, ausência de PII em `caplog.text`).

## Gestão de dados de teste

- Itens DynamoDB crus tipados (`{"S": …}`, `{"N": …}`, `{"BOOL": …}`) espelhando o schema real: Conversas do Contrato 5 (`PK=LEAD#<id>`, `SK=CONV#<session_id>`, `context`/`messages` como string JSON), Perfis (`SK=PROFILE` com `status`/`intent`/`area`) e alertas do Contrato 7 (`anomaly_id`, `features` JSON, `confidence`, `detected_at`).
- PII sintética (e-mail/telefone no `messages`) usada para provar que o payload do KPI e os logs não vazam texto bruto de mensagem.
- Nenhuma credencial real; env só via `monkeypatch.setenv` (`SESSIONS_TABLE`, `ALERTS_TABLE`, `CW_NAMESPACE`, `DASHBOARD_ALLOWED_ORIGIN`, `DASHBOARD_API_TOKEN`, `COGNITO_LOGIN_URL`) — nada hardcoded em código de produção.
- Sem arquivos em disco: toda a suíte é in-memory.
