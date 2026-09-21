# Code Summary — u7-dashboard

## Files created/modified

Todos os arquivos são novos, sob `apps/dashboard-api/` e `apps/dashboard-ui/` (nenhum app existente foi tocado):

| Arquivo | Papel |
|---------|-------|
| `apps/dashboard-api/handler.py` | Entry point Lambda (aws_proxy) do Contrato 2 — `GET /api/kpis`: 200 JSON, 401 sem bearer (CognitoAuthorizer na borda em produção), 404 path, 405 método, 500 em falha de agregação; wiring DI via `build_service` (boto3 preguiçoso, env: `SESSIONS_TABLE`, `ALERTS_TABLE`, `CW_NAMESPACE`, `CW_RESPONSE_METRIC`, `CW_COST_METRIC`, `DASHBOARD_ALLOWED_ORIGIN`); header CORS para o Community Cloud |
| `apps/dashboard-api/requirements.txt` | Dependência de produção (`boto3>=1.34`) |
| `apps/dashboard-api/infra/conversation_store.py` | Leitor do Contrato 5: dois scans paginados `begins_with(SK, CONV#)` (conversas) e `begins_with(SK, PROFILE)` (mapa `lead_id → perfil`); unmarshal de tipos/JSON; itens sem identificadores ignorados com log; erros → `ConversationStoreError` |
| `apps/dashboard-api/infra/alert_store.py` | Leitor do Contrato 7 (tabela `sdr-alerts`, dono U5): scan paginado, ordena `detected_at` desc, itens sem `anomaly_id` ignorados; `project_alert` projeta o alerta sem `features` (payload mínimo, PII-safe); erros → `AlertStoreError` |
| `apps/dashboard-api/infra/metrics.py` | `CloudWatchMeter` — `get_metric_data` para `response_time_p90` (stat p90) e `cost_monthly` (stat Maximum); datapoint ausente/erro de leitura/valor não numérico → `None` (degradação graciosa, zeros no snapshot) |
| `apps/dashboard-api/service/kpi_calc.py` | Funções puras determinísticas: `parse_iso8601`, `count_new_leads` (hoje/semana por `created_at`), `latest_conversation_by_lead`, `qualification_rate` (status `qualified` no perfil OU `context.lead_qualified` OU estado pós-qualificação), `state_funnel` (8 estados da esteira do u1 + `outros`), `scheduled_visits_count`, `intent_volume`, `route_distribution` (roleta u1: ≤ 500 m² → consultores, > 500 m² → diretor; override `route`/`route_target`/`assigned_to`; `parse_area_m2` tolera "1.200 m²"), `alerts_last_24h` |
| `apps/dashboard-api/service/kpis.py` | `KpiService.snapshot()` — orquestra conversas + perfis + alertas + meter → snapshot do Contrato 2 (`leads_today`, `leads_week`, `response_time_p90`, `qualification_rate`, `scheduled_visits`, `anomalies_count`, `cost_monthly`) + extensões aditivas (`generated_at`, `intents`, `route_distribution`, `funnel`, `alerts` cap 100); tabela vazia → zeros, item inválido → ignorado, falha de store → propaga (500); logs estruturados `log_event` (NFR5.1) |
| `apps/dashboard-api/tests/conftest.py` | Bootstrap de path (`sys.path` → `apps/dashboard-api`), padrão flat u1–u6 |
| `apps/dashboard-api/tests/unit/test_*.py` | 6 suítes unitárias (kpi_calc, conversation_store, alert_store, metrics, kpis, handler) |
| `apps/dashboard-api/tests/integration/test_kpis_pipeline.py` | Pipeline ponta a ponta: handler + FakeAwsClient (filtro de prefixo honrado) + fake boto3 em `sys.modules`; agregação completa, zeros em tabelas vazias, itens malformados, PII-safe no corpo e nos logs, store fora do ar → 500, CloudWatch fora do ar → 200 com zeros |
| `apps/dashboard-ui/app.py` | 1 página Streamlit Community Cloud: `fetch_kpis` (GET `/api/kpis` com http injetado, bearer de `DASHBOARD_API_TOKEN`, timeout, 401 → login Cognito, 500/erro → toast), cards FR7.2, esteira Kanban + gráficos de intenções/roleta FR7.3, tabela de alertas 24h FR7.4, callout Cognito `COGNITO_LOGIN_URL`; sem persistência local |
| `apps/dashboard-ui/requirements.txt` | Dependências de produção (`streamlit>=1.35`, `requests>=2.31`) |
| `apps/dashboard-ui/tests/conftest.py` + `tests/unit/test_app_smoke.py` | Smoke da UI: fetch_kpis (ok/401/500/rede/JSON inválido/bearer), `cognito_login_url`, render com `importorskip("streamlit")` |

## Key implementation decisions

- **KPI set = Contrato 2 + extensões aditivas**: os 7 campos contratuais (`leads_today`, `leads_week`, `response_time_p90`, `qualification_rate`, `scheduled_visits`, `anomalies_count`, `cost_monthly`) são calculados exatamente como especificados; `intents`, `route_distribution`, `funnel` e `alerts` entram como campos aditivos (política de versionamento: adição não-quebrante) para alimentar FR7.3/FR7.4 sem segunda chamada.
- **Leitor do Contrato 5 com dois scans**: mesmo padrão dos leitores do u5/u6 (`CONV#` + `PROFILE` sobre `sdr-sessions`); U7 é reader — nunca escreve. A qualificação (u1 não persiste `status="qualified"` no Lead) é derivada de três sinais tolerantes: `status` do perfil, `context.lead_qualified` da conversa mais recente ou estado pós-qualificação da esteira (`recommendation`/`scheduling`/`handoff`/`followup`).
- **Roleta (FR10.1/FR10.2) derivada da `area`**: u1 define `AREA_THRESHOLD_M2 = 500` com rodízio para consultores; como nenhum app grava ainda a rota auditada (FR10.3), a distribuição usa a regra do u1 sobre `area` com override por campo de rota quando existir — decisões documentadas como interpretação de POC.
- **Degradação graciosa vs. 500**: tabela vazia, item malformado (sem identificadores) ou datapoint de métrica ausente → zeros/estado vazio (nunca quebram o snapshot); falha de cliente DynamoDB → `ConversationStoreError`/`AlertStoreError` → 500 (Contrato 2); CloudWatch fora do ar → 200 com zeros (métrica operacional não deve derrubar o dashboard em POC).
- **Handler estilo Lambda aws_proxy, não FastAPI**: `.venv` não tem fastapi/flask/httpx — o boundary HTTP é a interface `handler(event, context)` do aws_proxy (mesmo padrão dos handlers u1–u6), testado por invocação direta com eventos proxy; autenticação fina fica no CognitoAuthorizer do API Gateway (fase de infra), com guarda de presença de bearer no handler.
- **PII-safe por projeção**: o snapshot agrega apenas contadores/estados; `messages` e `features` nunca saem dos readers — `project_alert` entrega só os campos do Contrato 7 relevantes à tabela de alertas; verificado ponta a ponta com PII sintética no corpo e no `caplog`.
- **UI thin e import-light**: toda a lógica fica no dashboard-api; `app.py` só busca, projeta e trata erro; `streamlit`/`requests` importados com preguiça (smoke roda sem streamlit instalado); login Cognito é placeholder por env (`DASHBOARD_API_TOKEN` no fetch, `COGNITO_LOGIN_URL` no botão) — o fluxo real é infra-stage.

## Test coverage summary

- Comando: `.venv/bin/python -m pytest apps/dashboard-api/tests apps/dashboard-ui/tests --cov=apps/dashboard-api --cov-report=term-missing --cov-fail-under=80`
- Resultado: **68 coletados — 68 passed, 2 skipped** (smoke de render com `streamlit` ausente), **TOTAL 98.40%** (piso 80% atendido, nenhuma meta relaxada).
- Componentes: `handler.py` 100%, `metrics.py` 100%, `kpi_calc.py` 98%, `kpis.py` 98%, `conversation_store.py` 94%, `alert_store.py` 91%.
- Suítes anteriores re-verificadas após a unidade: u1 = 50 passed, u2 = 58 passed, u3 = 86 passed, u4 = 50 passed, u5 = 51 passed + 1 skipped, u6 = 77 passed.

## Deviations from the plan

- **Sem FastAPI no boundary HTTP**: a diretriz admitia "fastapi or serverless-style handler"; como o `.venv` não tem fastapi/flask/httpx (verificado antes da geração), o Contrato 2 foi implementado como handler Lambda aws_proxy (`handler(event, context)`), mesmo padrão de todos os apps existentes — testável por invocação direta com eventos proxy e DI completa.
- **CloudWatch na POC**: `response_time_p90` e `cost_monthly` vêm de `CloudWatchMeter` (cliente boto3 injetado, nomes/namespace por env). Em POC a Lambda agrega `get_metric_data` (stat p90 para resposta, Maximum para custo); sem datapoint ou com CloudWatch fora do ar o snapshot degrada para zeros — nunca 500. Os emissores dessas métricas (instrumentação dos outros Lambdas) são infra-stage.
- **KPI set derivado**: além dos 7 campos do Contrato 2, o snapshot carrega `generated_at`, `intents`, `route_distribution`, `funnel` e `alerts` (aditivo, não-quebrante) para cumprir FR7.3/FR7.4 com uma única chamada da UI. `qualification_rate`/`scheduled_visits` derivam do estado da esteira do u1 porque o u1 não persiste status de qualificação no item do Lead.
- **Roleta sem dado gravado**: FR10.3 (rota auditada no DynamoDB) ainda não é materializado por nenhuma unidade anterior; a distribuição da roleta no dashboard deriva da regra do u1 sobre `area` (≤ 500 m² → consultores; > 500 m² → diretor), com override se um campo `route`/`route_target`/`assigned_to` existir no perfil.
- **UI/Cognito wiring**: login fica em placeholder configurável por env (`DASHBOARD_API_TOKEN` como bearer POC; `COGNITO_LOGIN_URL` para o botão de login) — o fluxo OAuth real com Cognito User Pools é infra-stage, conforme a diretriz; a UI não guarda nenhum dado local.
- **Streamlit import-guard**: `streamlit` não está no `.venv`; os testes de render usam `pytest.importorskip("streamlit")` (2 skipped aqui) e `app.py` importa `streamlit`/`requests` com preguiça para manter o smoke import-light.
- **Sem `__init__.py` nos `tests/`**: a suíte desta unidade roda `apps/dashboard-api/tests` e `apps/dashboard-ui/tests` num único comando; dois pacotes `tests` no mesmo pytest colidem (`ImportPathMismatchError`), então os diretórios de teste ficam sem `__init__.py` e a suíte de integração é self-contained (diferença em relação ao layout u1–u6, que roda uma suíte por comando).
- **CORS no handler**: o Community Cloud consome a API cross-origin; o header `Access-Control-Allow-Origin` é emitido pelo handler (env `DASHBOARD_ALLOWED_ORIGIN`, default `*`). Configuração de CORS no API Gateway fica para a fase de infra.
