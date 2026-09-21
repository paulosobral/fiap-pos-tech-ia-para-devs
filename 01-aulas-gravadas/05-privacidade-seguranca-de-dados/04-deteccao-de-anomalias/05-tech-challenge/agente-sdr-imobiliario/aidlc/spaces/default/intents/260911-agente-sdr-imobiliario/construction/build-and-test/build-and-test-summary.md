# Build and Test Summary — Agente SDR Imobiliário (POC)

## Status geral do build
- **BUILD OK** (`compileall -q apps` exit 0); smoke de import via suítes (conftest `sys.path`).
- Pré-requisitos atendidos: `.venv` Python 3.11+; sem serviços externos para build/test.

## Inventário de tipos de teste gerados
| Tipo | Arquivo de instrução | Status |
|---|---|---|
| Unit (por unit, do Code Generation) | `construction/*/code-generation/unit-test-instructions.md` | executado |
| Integration | `integration-test-instructions.md` | executado |
| Performance | `performance-test-instructions.md` | alvos e2e deferidos (`performance-validation`) |
| Security | `security-test-instructions.md` | suíte local executada; DAST deferido |

## Expectativa de cobertura por unit
Piso contratado 80%; obtido: u1 96.51 · u2 99.88 · u3 99.59 · u4 100.00 · u5 95.29 · u6 98.52 · u7-api 98.10 (ui smoke).

## Target Verification Matrix
Verdicts finais (sem `Pending`): `Met` = verificado nesta execução; `Unverified` = não executável localmente, com Owning Stage agendado no plano; `Not Met` = gap de feature (finding do gate).

| Target ID | Source | Expected | Actual | Evidence | Owning Stage | Verdict |
|---|---|---|---|---|---|---|
| T-COV-u1 | Testing Contract (80%) | ≥80% | 96.51% | `/tmp/opencode/bt-u1.log` | — | Met |
| T-COV-u2 | idem | ≥80% | 99.88% | `bt-u2.log` | — | Met |
| T-COV-u3 | idem | ≥80% | 99.59% | `bt-u3.log` | — | Met |
| T-COV-u4 | idem | ≥80% | 100.00% | `bt-u4.log` | — | Met |
| T-COV-u5 | idem | ≥80% | 95.29% | `bt-u5.log` | — | Met |
| T-COV-u6 | idem | ≥80% | 98.52% | `bt-u6.log` | — | Met |
| T-COV-u7 | idem (api; ui smoke) | ≥80% api | 98.10% api / 8+2 smoke | `bt-u7-api.log`, `bt-u7-ui.log` | — | Met |
| T-TESTS-PER-COMP | Testing Contract | 5–8+ por componente | 71–117 por unit | suítes | — | Met |
| T-CI-MERGE | Testing Contract | execução em CI antes do merge | Pending no pipeline local | este estágio | **ci-pipeline** | Unverified |
| NFR2.1 | requirements | logs JSON | log_event em todas as units (caplog) | suítes | — | Met |
| NFR2.2 | requirements (traceability u2) | conforme design | OK por traceability + suítes | suítes | — | Met |
| NFR2.3 | requirements | consent na 1ª msg | consent gate em `/start` + testes | test_sales_flow/test_entities | — | Met |
| NFR2.4 | requirements | trilha auditoria de leads | mínimo POC: logs JSON + session store auditável | suítes u1/u4 | — | Met (mínimo; gap append-only listado no gate) |
| NFR2.5 | requirements | PII-safe | caplog sem PII | suítes | — | Met |
| NFR3.1 | requirements | PII mascarada antes do LLM | security_layer mask | test_pii_masker | — | Met |
| NFR3.2 | requirements | consent registrado | store de consent | test_session_store | — | Met |
| NFR3.3 | requirements | TTL de retenção | TTL em stores u1/u4/u5 | test_session_store/test_conversation_store | — | Met |
| NFR4.1 | requirements | DLQ EventBridge | config IaC ausente (POC local) | traceability u4/u5/u7 | **deployment-pipeline** | Unverified |
| NFR4.2 | requirements (traceability u1) | conforme design | OK | handler u1 | — | Met |
| NFR5.1 | requirements | logs JSON por serviço | log_event u2–u7 | suítes | — | Met |
| NFR5.2 | requirements | traces distribuídos | sem OpenTelemetry (POC) | grep | (novo work) | Not Met |
| NFR5.3 | requirements | métricas de negócio | metrics.py + KPIs API | u7 suítes | — | Met |
| NFR6.1 | requirements | custo ~R$15/mês | exige tráfego real | — | **observability-setup** | Unverified |
| NFR6.2 | requirements | limite de tokens/monitor | sem client LLM real no POC | grep | **observability-setup** | Unverified |
| NFR7.1 | requirements | guardrails/denied topics | test_guardrails | suítes u1 | — | Met |
| NFR7.2 | requirements | prompt injection | test_guardrails | suítes u1 | — | Met |
| NFR7.3 | requirements | evasão de PII | test_guardrails | suítes u1 | — | Met |
| NFR8.1 | requirements | auto scaling | infra cloud ausente (POC) | — | **deployment-pipeline** | Unverified |
| NFR9.1 | requirements | Terraform por serviço | sem `.tf` (POC) | — | **deployment-pipeline** | Unverified |
| NFR9.2 | requirements | start.sh build/apply | ausente (POC) | — | **deployment-pipeline** | Unverified |
| NFR9.3 | requirements | stop.sh destroy | ausente (POC) | — | **deployment-pipeline** | Unverified |
| NFR9.4 | requirements | ambiente recriável | ausente (POC) | — | **deployment-pipeline** | Unverified |
| FR1.4 | requirements | botões inline | não implementado (texto livre OK) | grep | (novo work) | Not Met |
| FR2.3 | requirements | B2B vs investidor PF | questionário único (Deferred u1) | traceability u1 | (novo work) | Not Met |
| FR4.1 | requirements | índice FAISS em memória | interface `properties_rag` injetável; índice real ausente | sales_flow | (novo work) | Not Met |
| FR4.2 | requirements | top-k compatível | busca com filtros | sales_flow + suítes | — | Met |
| FR4.3 | requirements | nunca inventar imóveis | constraint na busca | sales_flow + suítes | — | Met |
| FR5.1 | requirements | validar data/hora | validação + scheduler | sales_flow + suítes | — | Met |
| FR5.2 | requirements | compromisso no calendário simulado | `scheduler` injetável grava appointment | sales_flow | — | Met |
| FR5.3 | requirements | convite ICS | não implementado | grep | (novo work) | Not Met |
| FR5.4 | requirements | notificar corretor (canal interno) | canal textual simulado (POC) | sales_flow | — | Met |
| FR6.1 | requirements | handoff markdown | handoff_builder | sales_flow + suítes | — | Met |
| FR6.2 | requirements | mensagem + arquivo ao corretor | mensagem/resumo sim; arquivo não | sales_flow | (novo work) | Not Met (parcial) |
| FR6.3 | requirements | unmask PII no handoff | security_layer unmask | test_pii_masker | — | Met |
| FR9.1 | requirements | features de comportamento | feature_extractor real | u5 suítes | — | Met |
| FR9.2 | requirements | IF+PCA+Autoencoder | IF + PCA (erro reconstrução) | scorer.py; deviation em u5/code-summary | (deviation aceita no gate anterior) | Not Met (deviation) |
| FR9.3 | requirements | alertas | alert_store | u5 suítes | — | Met |
| FR9.4 | requirements | restrição agendamento | scheduler_gate + contrato U1 | u5 integration | — | Met |
| FR11.1 | requirements | demo MCP HubSpot ao vivo | requer ambiente+credenciais | — | **deployment-execution** | Unverified |

## Readiness assessment
- **build-ready**: ✓ (compileall + suítes verdes)
- **test-ready**: ✓ (unit/integration 624 passed / 0 failed; segurança executável local Met)
- **deployment-ready**: ✗ (NFR4.1/NFR8.1/NFR9.x dependem de `deployment-pipeline`; é o escopo previsto do plano)

## Known limitations / outstanding items (surfaced no gate)
1. Gaps de feature **Not Met**: FR1.4, FR2.3, FR4.1, FR5.3, FR6.2 (parcial), NFR5.2, FR9.2 (deviation aceita) — detalhes e estimativas em `cross-unit-traceability.md`.
2. Deferreds com owning stage válido: NFR1.1/1.2, NFR4.1, NFR6.1/6.2, NFR8.1, NFR9.1–9.4, FR11.1, T-CI-MERGE.
3. NFR2.4 implementado como mínimo POC (logs + session store); trilha append-only dedicada é evolução possível (~2–4 h).
