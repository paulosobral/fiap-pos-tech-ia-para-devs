# Cross-Unit Traceability — Build and Test (gate de cobertura final)

Fontes: `inception/requirements-analysis/requirements.md` (12 FRs / 9 NFRs, folhas `FRx.y`/`NFRx.y`), `inception/user-stories/stories.md` (103 ACs), `construction/*/code-generation/traceability.json` (7 units), verificação real desta execução.

**Verdict: PARTIAL (reavaliado em 2026-10-08) — gaps de feature restantes: FR1.4, FR2.3, FR5.3 (parcial), FR6.2 (parcial), NFR5.2, mais 1 deviation (FR9.2) e itens Partial (NFR4.1, NFR5.3, NFR6.2, NFR8.1). Resolvidos desde 2026-09-21: FR4.1, FR11.1, NFR9.1–9.4. ACs herdam por grupo.**

## Cobertura por folha FR (status da execução local)
| ID | Unit | Status | Target / evidência |
|---|---|---|---|
| FR1.1, FR1.2, FR1.3 | u1 (+u2 voz) | OK | handler/sales_flow/voice-adapter + suítes |
| FR1.4 | — | **GAP** | botões inline não implementados (texto livre aceito) |
| FR2.1, FR2.2 | u1 | OK | intent + coleta de estrutura (sales_flow/lead_qualifier) |
| FR2.3 | u1 | **GAP** (Deferred) | B2B vs investidor PF: questionário adaptativo por perfil não implementado |
| FR3.1, FR3.2, FR3.3 | u1 | OK | lead_qualifier + sales_flow + testes |
| FR4.2, FR4.3 | u1 | OK | busca top-k e constraint anti-alucinação via `properties_rag` injetável |
| FR4.1 | u1 | OK | `faiss-cpu` + TF-IDF em `properties_catalog.py`; ressalva: índice reconstruído a cada busca |
| FR5.1, FR5.2 | u1 | OK | validação de data + gravação de compromisso via `scheduler` injetável |
| FR5.3 | u1 | **GAP (parcial)** | `_build_ics` gera o convite (`state["ics_invite"]`), mas o handler não o envia |
| FR5.4 | u1 | OK | notificação ao corretor (canal textual simulado no POC) |
| FR6.1, FR6.3 | u1 | OK | handoff markdown + unmask PII (security_layer) |
| FR6.2 | u1 | **GAP (parcial)** | mensagem/resumo existe; **arquivo** de resumo não |
| FR7.1–FR7.4 | u7 | OK | dashboard-api + dashboard-ui (KPIs, gráficos, roleta, anomalias); acrescidos: tabela de leads com contato, botão HubSpot, sessão persistente, identidade visual (ADR-016/018); widget de custo removido |
| FR8.1–FR8.3 | u7/u1 | OK | (métricas/monitoramento de leads conforme traceability u7) |
| FR9.1, FR9.3, FR9.4 | u5 | OK | feature_extractor, alert_store, scheduler_gate |
| FR9.2 | u5 | **DEVIAÇÃO ACEITA** | Isolation Forest + PCA (erro de reconstrução estilo autoencoder) substitui Autoencoder; justificativa em `u5/code-summary.md` §Deviations; gate code-generation aprovado |
| FR10.1–FR10.3 | u7 | OK | (anomalias no dashboard conforme traceability u7) |
| FR11.1 | u3 | OK | HubSpot via MCP remoto validado ao vivo (2026-10-08): contato criado/atualizado e lead real do Telegram; `scripts/hubspot_authorize.py` |
| FR11.2, FR11.3 | u3 | OK | adapter CRM + masking |
| FR12.1–FR12.3 | u4 | OK | ingest + session store + TTL |

## Cobertura por folha NFR (status da execução local)
| ID | Unit | Status | Target / evidência |
|---|---|---|---|
| NFR1.1, NFR1.2 | — | **Unverified** | p50/p90 com LLM real → owner `performance-validation` |
| NFR2.1 | todas | OK | `log_event` JSON por serviço (caplog provado) |
| NFR2.2 | u2/u3 | OK | traceability u2/u3 |
| NFR2.3, NFR2.5, NFR3.1, NFR3.2, NFR3.3 | u1/u4/u5 | OK | consentimento explícito (ADR-017), PII-safe, máscara antes da LLM só para e-mail/telefone/CNPJ (nome não é mascarado, decisão de produto), TTL |
| NFR2.4 | u4/u1 | OK (mínimo POC) | logs JSON por operação + session store auditável; trilha append-only dedicada não existe |
| NFR4.1 | u2/u3/u4/u5/u6/u7 | **Partial** | DLQs SQS existem; EventBridge sem `dead_letter_config` → owner `deployment-pipeline` |
| NFR4.2 | u1 | OK | traceability u1 (handler) |
| NFR5.1 | u2–u7 | OK | logs JSON por serviço |
| NFR5.2 | — | **GAP** | traces distribuídos (OpenTelemetry) não implementados |
| NFR5.3 | u7 | **Partial** | KPIs de negócio OK; `ResponseTimeP90`/custo lidos do CloudWatch mas nunca emitidos (sempre 0); widget de custo removido |
| NFR6.1 | — | **Unverified** | custo real exige tráfego → owner `observability-setup` |
| NFR6.2 | u1 | **Partial** | `max_tokens` por chamada; sem monitor/alarme → owner `observability-setup` |
| NFR7.1, NFR7.2, NFR7.3 | u1 | OK | guardrails/injection/PII evasion (`test_guardrails.py`) |
| NFR8.1 | — | **Partial** | Lambdas nativas + ECS com scheduled scaling; sem teste de carga → owner `performance-validation` |
| NFR9.1–NFR9.4 | — | OK | 21 `.tf`, `start.sh`, `stop.sh`, ambiente recriado várias vezes |

## ACs (103) — herança por grupo
Cada story pertence a um grupo que mapeia FRs; os ACs herdam o status dos FRs do grupo (ACs não possuem target de código próprio no POC):
| Grupo (stories.md) | ACs | FRs do grupo | Status herdado |
|---|---|---|---|
| 1 Atendimento Conversacional | 16 | FR1.1–1.3 | OK (exceto FR1.4 → GAP) |
| 2 Identificação de Intenção | 9 | FR2.1–2.3 | OK (FR2.3 GAP) |
| 3 Qualificação de Leads | 8 | FR3 | OK |
| 4 RAG Imóveis | 10 | FR4 | OK |
| 5 Agendamento | 6 | FR5 | OK (FR5.3 parcial) |
| 6 Resumo/Handoff | 6 | FR6 | OK (FR6.2 parcial) |
| 7 Dashboard | 5 | FR7 | OK |
| 8 Anomalias | 4 | FR9.3–9.4 | OK |
| 9 CRM HubSpot | 6 | FR11 | OK |
| 10 NFR Coverage | AC10–AC16 (27) | NFR1–NFR9 | misto (ver tabela NFR) |
| Nice-to-have | 10 | FR8/FR10 | OK conforme traceability u7 |

## Elementos não cobertos (findings para o gate)
1. **FR1.4** — botões inline (fix estimado: ~2–3 h)
2. **FR2.3** — B2B vs investidor PF (fix estimado: ~3–5 h)
3. **FR5.3** — enviar o `.ics` já gerado ao lead/corretor (fix estimado: ~1 h)
4. **NFR5.3** — emitir `put_metric_data` de latência/custo (fix estimado: ~2–3 h) ou remover a leitura
5. **FR6.2** — arquivo de resumo ao corretor (fix estimado: ~1–2 h)
6. **NFR5.2** — traces distribuídos (fix estimado: ~4–6 h)
7. **FR9.2** — Autoencoder real (deviation aceita no gate anterior; fix estimado: ~6–10 h)
8. **NFR2.4** — trilha append-only dedicada (mínimo POC existente; fix estimado: ~2–4 h)
9. Deferreds com owner válido no plano: NFR1.1, NFR1.2, NFR8.1 (performance-validation), NFR4.1 (deployment-pipeline), NFR6.1, NFR6.2 (observability-setup), contrato CI (ci-pipeline)
10. Observação de otimização (não é requisito): o índice FAISS é reconstruído a cada busca e a varredura de leads no DynamoDB é `Scan` (gatilho de reavaliação ~5 mil leads, ADR-015/016).
