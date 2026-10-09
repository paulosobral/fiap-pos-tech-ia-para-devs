# Build and Test Summary — Agente SDR Imobiliário (POC)

## Status geral do build
- **BUILD OK** (`compileall -q apps` exit 0; `terraform validate` OK; pacote do crm-adapter 27,7 MB). Última verificação: 2026-10-08.
- Pré-requisitos atendidos: `.venv` Python 3.11+; sem serviços externos para build/unit; o gate com LLM real exige chave (ver `build-instructions.md`).

## Inventário de tipos de teste gerados
| Tipo | Arquivo de instrução | Status |
|---|---|---|
| Unit (por unit, do Code Generation) | `construction/*/code-generation/unit-test-instructions.md` | executado |
| Integration | `integration-test-instructions.md` | executado |
| Quality gate (LLM real, TaaC) | `apps/conversation-router/tests/quality/` | executado: 40/40 (inclui anomalias ponta a ponta) |
| Performance | `performance-test-instructions.md` | alvos e2e deferidos (`performance-validation`) |
| Security | `security-test-instructions.md` | suíte local executada; DAST deferido |

## Expectativa de cobertura por unit
Piso contratado 80%; obtido (2026-10-08): u1 88.70 · u2 99.38 · u3 99.13 · u4 100.00 · u5 99.01 · u6 98.52 · u7-api 97.70 · u7-ui 80.32 (no limite). u1 caiu de 96.51 porque o código cresceu (roteador tool-agent, fotos, contato) com parte do caminho exercitada só pelo gate com LLM real.

## Target Verification Matrix
Verdicts finais (sem `Pending`): `Met` = verificado nesta execução; `Unverified` = não executável localmente, com Owning Stage agendado no plano; `Not Met` = gap de feature (finding do gate).

| Target ID | Source | Expected | Actual | Evidence | Owning Stage | Verdict |
|---|---|---|---|---|---|---|
| T-COV-u1 | Testing Contract (80%) | ≥80% | 88.70% | pytest --cov (ver `test-results.md`) | — | Met |
| T-COV-u2 | idem | ≥80% | 99.38% | pytest --cov u2 | — | Met |
| T-COV-u3 | idem | ≥80% | 99.13% | pytest --cov u3 | — | Met |
| T-COV-u4 | idem | ≥80% | 100.00% | pytest --cov u4 | — | Met |
| T-COV-u5 | idem | ≥80% | 99.01% | pytest --cov u5 | — | Met |
| T-COV-u6 | idem | ≥80% | 98.52% | pytest --cov u6 | — | Met |
| T-COV-u7 | idem (api e ui) | ≥80% | 97.70% api / 80.32% ui | pytest --cov u7 | — | Met (ui no limite) |
| T-QUALITY-GATE | Plano de naturalidade §5 / ADR-019 | gate LLM real verde antes do deploy | 40/40 em 7 min 21 s | `test-results.md` | — | Met |
| T-TESTS-PER-COMP | Testing Contract | 5–8+ por componente | 38–440 por unit (999 no total) | suítes | — | Met |
| T-CI-MERGE | Testing Contract | execução em CI antes do merge | Pending no pipeline local | este estágio | **ci-pipeline** | Unverified |
| NFR2.1 | requirements | logs JSON | log_event em todas as units (caplog) | suítes | — | Met |
| NFR2.2 | requirements (traceability u2) | conforme design | OK por traceability + suítes | suítes | — | Met |
| NFR2.3 | requirements | consent na 1ª msg | consentimento explícito (só "sim" claro), reperguntado caso contrário (ADR-017) | test_conversation_polish/test_sales_flow | — | Met |
| NFR2.4 | requirements | trilha auditoria de leads | mínimo POC: logs JSON + session store auditável | suítes u1/u4 | — | Met (mínimo; gap append-only listado no gate) |
| NFR2.5 | requirements | PII-safe | caplog sem PII | suítes | — | Met |
| NFR3.1 | requirements | PII mascarada antes do LLM | e-mail/telefone/CNPJ mascarados; nome NÃO (decisão de produto: vem do perfil do Telegram); saída checada por `check_output_leak` | test_pii_masker | — | Met (escopo reduzido por decisão) |
| NFR3.2 | requirements | consent registrado | store de consent | test_session_store | — | Met |
| NFR3.3 | requirements | TTL de retenção | TTL em stores u1/u4/u5 | test_session_store/test_conversation_store | — | Met |
| NFR4.1 | requirements | DLQ EventBridge | DLQs SQS existem (voice/crm/ingest, `infra/sqs.tf`); EventBridge sem `dead_letter_config` | `infra/sqs.tf` | **deployment-pipeline** | Partial |
| NFR4.2 | requirements (traceability u1) | conforme design | OK | handler u1 | — | Met |
| NFR5.1 | requirements | logs JSON por serviço | log_event u2–u7 | suítes | — | Met |
| NFR5.2 | requirements | traces distribuídos | sem X-Ray/OpenTelemetry (sem `tracing_config` nos `.tf`) | grep | (novo work) | Not Met |
| NFR5.3 | requirements | métricas de negócio | contagens/funil/anomalias vêm das tabelas (OK); `ResponseTimeP90` e custo são LIDOS do CloudWatch mas **nenhum componente emite** (`put_metric_data` ausente) → sempre 0; o widget de custo foi removido do dashboard (ADR-016) | u7 suítes + grep | (novo work) | Partial |
| NFR6.1 | requirements | custo ~R$15/mês | exige tráfego real | — | **observability-setup** | Unverified |
| NFR6.2 | requirements | limite de tokens/monitor | `max_tokens` por chamada (40/300 no router); sem monitor/alarme de gasto | `service/llm.py` | **observability-setup** | Partial |
| NFR7.1 | requirements | guardrails/denied topics | test_guardrails | suítes u1 | — | Met |
| NFR7.2 | requirements | prompt injection | test_guardrails | suítes u1 | — | Met |
| NFR7.3 | requirements | evasão de PII | test_guardrails | suítes u1 | — | Met |
| NFR8.1 | requirements | auto scaling | Lambdas escalam nativamente; ECS com `aws_appautoscaling_scheduled_action` (janela 09–18h BRT); sem teste de carga | `infra/ecs.tf` | **performance-validation** | Partial |
| NFR9.1 | requirements | Terraform por serviço | 21 arquivos `infra/*.tf`, `terraform validate` OK, aplicado várias vezes pelo `start.sh` | `infra/` | — | Met |
| NFR9.2 | requirements | start.sh build/apply | `start.sh` (testes, gate LLM, build, 2 applies, smoke, webhook) | `start.sh` | — | Met |
| NFR9.3 | requirements | stop.sh destroy | `stop.sh` (destroy + limpeza de log groups; salva o refresh token do HubSpot antes) | `stop.sh` | — | Met |
| NFR9.4 | requirements | ambiente recriável | destruído e recriado repetidas vezes na prática | uso real | — | Met |
| FR1.4 | requirements | botões inline | não implementado (texto livre OK) | grep | (novo work) | Not Met |
| FR2.3 | requirements | B2B vs investidor PF | questionário único (Deferred u1) | traceability u1 | (novo work) | Not Met |
| FR4.1 | requirements | índice FAISS em memória | `faiss-cpu` + TF-IDF em `properties_catalog.py` (índice reconstruído a cada busca — otimização pendente) | properties_catalog | — | Met |
| FR4.2 | requirements | top-k compatível | busca com filtros | sales_flow + suítes | — | Met |
| FR4.3 | requirements | nunca inventar imóveis | constraint na busca | sales_flow + suítes | — | Met |
| FR5.1 | requirements | validar data/hora | validação + scheduler | sales_flow + suítes | — | Met |
| FR5.2 | requirements | compromisso no calendário simulado | `scheduler` injetável grava appointment | sales_flow | — | Met |
| FR5.3 | requirements | convite ICS | `_build_ics` gera o `.ics` (RFC 5545) e guarda em `state["ics_invite"]`, mas o handler **não o envia** ao lead/corretor | sales_flow, grep handler | (novo work) | Not Met (parcial) |
| FR5.4 | requirements | notificar corretor (canal interno) | canal textual simulado (POC) | sales_flow | — | Met |
| FR6.1 | requirements | handoff markdown | handoff_builder | sales_flow + suítes | — | Met |
| FR6.2 | requirements | mensagem + arquivo ao corretor | mensagem/resumo sim; arquivo não | sales_flow | (novo work) | Not Met (parcial) |
| FR6.3 | requirements | unmask PII no handoff | security_layer unmask | test_pii_masker | — | Met |
| FR9.1 | requirements | features de comportamento | feature_extractor real | u5 suítes | — | Met |
| FR9.2 | requirements | IF+PCA+Autoencoder | IF + PCA (erro reconstrução) | scorer.py; deviation em u5/code-summary | (deviation aceita no gate anterior) | Not Met (deviation) |
| FR9.3 | requirements | alertas | alert_store | u5 suítes | — | Met |
| FR9.4 | requirements | restrição agendamento | scheduler_gate + contrato U1 | u5 integration | — | Met |
| FR11.1 | requirements | demo MCP HubSpot ao vivo | `HubSpotCrmGateway` via MCP remoto (OAuth 2.1+PKCE); contato criado/achado/atualizado e lead real do Telegram no HubSpot (2026-10-08) | `test_hubspot_mcp.py` + validação manual | — | Met |
| FR7.x (leads) | PRD §10 / ADR-016 | tabela de leads com contato + botão HubSpot | `GET /api/leads`, `POST /api/leads/{id}/crm`, tabela no dashboard | `test_leads.py`, `test_app_smoke.py` | — | Met (sem teste de navegador) |
| NFR-SESSION | ADR-016 | sessão sobrevive ao F5 | cookie opaco + cofre no servidor + refresh | `TestSessionVault` | — | Met (sem teste de navegador) |

## Readiness assessment
- **build-ready**: ✓ (compileall + suítes verdes)
- **test-ready**: ✓ (999 unit/integration passed / 0 failed; gate LLM real 40/40; segurança executável local Met)
- **deployment-ready**: ✓ na prática (infra aplicada e recriada várias vezes); formalmente pendentes `deployment-pipeline` (NFR4.1 EventBridge DLQ) e `ci-pipeline` (T-CI-MERGE)

## Known limitations / outstanding items (surfaced no gate)
1. Gaps de feature **Not Met**: FR1.4, FR2.3, FR5.3 (convite gerado mas não enviado), FR6.2 (parcial), NFR5.2, FR9.2 (deviation aceita); **Partial**: NFR5.3 (métricas de latência/custo nunca emitidas), NFR6.2, NFR4.1, NFR8.1 — detalhes em `cross-unit-traceability.md`.
2. Deferreds com owning stage válido: NFR1.1/1.2, NFR6.1/6.2, NFR8.1, T-CI-MERGE.
4. Mudanças desde a última execução (2026-09-21): ADRs 014–019 (Cognito, NL pela LLM, dashboard com leads/sessão, consentimento explícito, identidade visual, gate de anomalias), HubSpot via MCP, voz tratada como texto digitado, fotos via crawler. Registradas em `inception/domain-design/decisions.md`.
3. NFR2.4 implementado como mínimo POC (logs + session store); trilha append-only dedicada é evolução possível (~2–4 h).
