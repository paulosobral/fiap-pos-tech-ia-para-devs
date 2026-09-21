# Code Summary — u5-anomaly

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/anomaly-detector/`, estrutura do PRD §7.4):**
- `handler.py` — handler EventBridge (wiring de produção com boto3/env e DI por construtor); clock injetável (`utc_now`); scorer configurável via `ANOMALY_SCORER` (`heuristic` default | `sklearn`) e `ANOMALY_THRESHOLD`; payload inesperado do scheduler nunca impede o job diário
- `service/anomaly_detector.py` — orquestrador AnomalyDetector (FR9): ler conversas (Contrato 6) → extrair features → score → salvar alerta (Contrato 7) → restringir agendamento (FR9.4); `anomaly_id` determinístico `session_id#data-do-job` (reprocesso idempotente via put overwrite); falha de leitura/scoring propaga para retry async da Lambda, falha pontual de extração é contada (`errors`) sem derrubar o job; logs estruturados JSON PII-safe (`log_event`) (NFR5.1 + NFR2.1)
- `service/feature_extractor.py` — ConversationFeatureExtractor (FR9.1): volume, comprimento médio, sentimento heurístico (palavras negativas pt-BR) e razão de horários atípicos (fora de 08:00–18:59); função pura e determinística, tolerante a mensagens malformadas (não-dict, sem texto, timestamp inválido)
- `service/scorer.py` — Protocol `AnomalyScorer` + `HeuristicScorer` (combinação ponderada determinística 100% stdlib, threshold default 0.7) + `SklearnScorer` (Isolation Forest + PCA com erro de reconstrução como sinal residual estilo autoencoder; imports guardados dentro do método → `ScorerDependencyError` com mensagem clara se scikit-learn ausente; lote < 5 cai para o fallback heurístico — evita falso-positivo de fit não-supervisionado em amostra mínima)
- `service/scheduler_gate.py` — SchedulingGate (FR9.4): materializa a restrição no item da anomalia (`scheduling_restricted=True`, `action_taken="schedule_restricted"`) e expõe checks consultáveis `is_restricted(lead_id)` / `restriction_reason(lead_id)` para o Scheduler (u1) e Build-and-Test
- `infra/conversation_store.py` — ConversationStore (leitor do Contrato 6): scan paginado com filtro `begins_with(SK, CONV#)`, unmarshal de tipos DynamoDB (N/S/BOOL) e decodificação JSON de `messages`/`context`; itens sem `session_id`/`lead_id` ignorados com log; erros → `ConversationStoreError`
- `infra/alert_store.py` — AlertStore (owner do Contrato 7): put de anomalias com `PK=anomaly_id` (schema do contrato: `anomaly_id`, `lead_id`, `features`, `confidence`, `type`, `detected_at`, `status`, `action_taken`), update de restrição, consulta de restrição aberta por lead via GSI `lead-index`; erros → `AlertStoreError`

**Testes:** `tests/unit/test_feature_extractor.py` (8), `test_scorer.py` (8), `test_anomaly_detector.py` (8), `test_conversation_store.py` (6), `test_alert_store.py` (7), `test_scheduler_gate.py` (5), `tests/integration/test_anomaly_pipeline.py` (8)

**Config:** `apps/anomaly-detector/requirements.txt` (`boto3>=1.34`), `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz, `apps/{conversation-router,voice-adapter,crm-adapter,contact-ingest}/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- DI por construtor em todos os componentes (store, extractor, scorer, gate) — a suíte roda sem AWS real, sem rede e sem dependências além do `.venv` existente; o evento EventBridge é JSON puro.
- **Stack ML realista (FR9.2)**: o `.venv` do repo NÃO tem scikit-learn/numpy (verificado: `import sklearn` → `ModuleNotFoundError`). Por isso o scorer vive atrás do Protocol `AnomalyScorer`: `SklearnScorer` implementa o ensemble contratado com imports guardados (erro claro `ScorerDependencyError` em runtime sem a dependência) e `HeuristicScorer` é o fallback determinístico que a suíte executa ponta a ponta. A troca é feita por env (`ANOMALY_SCORER`), sem tocar código.
- **Postura de falso-positivo documentada** (constraint da unidade): thresholds conservadores por feature no fallback; lote < `MIN_BATCH` (5) usa heurística em vez de fit não-supervisionado em amostra mínima; kind/type da anomalia indica a feature dominante (`high_volume`, `long_messages`, `negative_sentiment`, `atypical_hours`) ou `ml_ensemble` no modo sklearn — rastreável no dashboard (U7) para validação.
- **Idempotência do job diário**: `anomaly_id = session_id#YYYY-MM-DD` determinístico — reprocesso do mesmo dia sobrescreve o mesmo item (put), sem alertas duplicados; `detected_at` usa o clock injetável.
- **Restrição de agendamento (FR9.4) na tabela de alertas**: U5 é owner apenas dos alertas (Contrato 7); a tabela de sessões é do u1. A restrição é materializada no item da anomalia e consultável por GSI `lead-index` — o Scheduler (u1) consome via `SchedulingGate.is_restricted(lead_id)` sem acoplamento direto.
- PII-safe em profundidade (NFR2.1): textos de mensagens, e-mails e telefones nunca entram em logs — apenas `lead_id`, `session_id`/`anomaly_id`, `type` e `confidence`; testes de integração provam via `caplog`.
- Leitura por scan paginado com filtro de prefixo — Contrato 6 não define GSI de varredura diária; custo/dimensionamento do scan é item para Build and Test.
- Logs estruturados JSON (`log_event`) no mesmo estilo do u2/u3/u4 (NFR5.1).

## Test coverage summary

- Comando: `.venv/bin/python -m pytest apps/anomaly-detector/tests --cov=apps/anomaly-detector --cov-report=term-missing --cov-fail-under=80`
- Resultado: **52 coletados — 51 passed, 1 skipped** (ensemble real pula sem sklearn), **TOTAL 94.53%** (piso 80% atendido).
- Componentes: `anomaly_detector.py` 98–100%, `feature_extractor.py` 98%, `conversation_store.py` 93%, `alert_store.py` 92%, `scheduler_gate.py` 96%, `handler.py` 97%, `scorer.py` 70% (linhas do ensemble sklearn executam apenas com a dependência instalada — caminhos executáveis sem sklearn 100% cobertos; ver Deviations).
- Suítes anteriores re-verificadas após a unidade: u1 = 50 passed, u2 = 58 passed, u3 = 86 passed, u4 = 50 passed.

## Deviations from the plan

- **scikit-learn/numpy indisponíveis no `.venv`** (verificado: `ModuleNotFoundError`): `SklearnScorer` ficou com imports guardados dentro do método (erro claro em runtime sem a dependência) e `HeuristicScorer` (stdlib) é o fallback executável que a suíte exercita ponta a ponta. Consequência de cobertura: as linhas do ensemble sklearn só medem em ambiente com scikit-learn (teste `importorskip`), deixando `scorer.py` em 70% — o piso contratado é sobre a unidade (94.53% TOTAL) e nenhuma meta foi relaxada.
- **Autoencoder real não implementado na POC**: o terceiro modelo contratado (FR9.2) é representado pelo erro de reconstrução da PCA (sinal residual equivalente em espírito), dentro do ensemble Isolation Forest + PCA — decisão documentada para validação de falsos positivos; substituição por autoencoder (Keras/PyTorch) não muda o Protocol.
- **Mecanismo da restrição de agendamento (FR9.4)**: o Contrato 7 não definia como o Scheduler consultaria a restrição; implementado como flag no item da anomalia (`scheduling_restricted`, `restriction_reason`) + consulta por GSI `lead-index` (`SchedulingGate.is_restricted`). U5 não escreve na tabela de sessões (owner U1).
- **`anomaly_id` determinístico** (`session_id#data-do-job`): o Contrato 7 não especificava estratégia de id; necessário para idempotência do reprocesso diário (put overwrite em vez de alertas duplicados).
- **Campos aditivos no item de alerta** (`scheduling_restricted`, `restriction_reason`) além do schema mínimo do Contrato 7 — aditivos, sem quebra do schema compartilhado (política de versioning da POC).
- **Leitura de conversas via scan** com filtro `begins_with(SK, CONV#)`: o Contrato 6 não prevê GSI para varredura diária; aceito na POC, otimização (GSI dedicada ou export) fica para Build and Test.
- **boto3 ausente no `.venv`**: o wiring do `handler()` é testado com módulo fake injetado em `sys.modules` (padrão já usado pelo u4).
- IaC (Terraform — regra do EventBridge, tabela de alertas) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
