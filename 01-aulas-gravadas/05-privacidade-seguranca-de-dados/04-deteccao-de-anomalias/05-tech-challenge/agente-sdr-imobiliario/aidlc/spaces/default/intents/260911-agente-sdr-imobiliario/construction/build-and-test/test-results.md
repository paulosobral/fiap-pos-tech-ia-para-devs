# Test Results — Build and Test (run de 2026-10-08)

## Build status
- **compileall `apps`**: exit 0 — BUILD OK.
- `terraform validate` (infra/): Success. `bash -n start.sh stop.sh`: OK.
- Pacote do `crm-adapter` (com `mcp`): 27,7 MB compactado (< 50 MB).

## Resultados por unit (unitários + integração; sem o gate LLM)
| Unit | Passed | Failed | Skipped | Cobertura | Piso 80% |
|---|---|---|---|---|---|
| u1 conversation-router | 460 | 0 | 0 | 88,54% (600 linhas sem cobertura de 5237) | ✓ |
| u2 voice-adapter | 75 | 0 | 0 | 99,38% | ✓ |
| u3 crm-adapter | 129 | 0 | 0 | 99,13% | ✓ |
| u4 contact-ingest | 71 | 0 | 0 | 100,00% | ✓ |
| u5 anomaly-detector | 87 | 0 | 0 | 99,01% | ✓ |
| u6 followup | 87 | 0 | 0 | 98,52% | ✓ |
| u7 dashboard-api | 72 | 0 | 0 | 97,70% | ✓ |
| u7 dashboard-ui | 38 | 0 | 0 | 80,32% (no limite) | ✓ |

**Total: 1019 passed, 0 failed, 0 skipped** (run anterior, 2026-09-21: 624 passed, 5 skipped).

## Gate de qualidade com LLM real (`apps/conversation-router/tests/quality`)
- **48 passed, 0 failed** em 6 min 39 s (re-executado após a ADR-020) (OpenRouter real, mesmo tiering de produção).
- Cobre: resolução de referências/ordinais, navegação, listas, fotos, contato/telefone (inclusive falado), fechamento de lead ponta a ponta, tipo de imóvel diferente do pedido sem narrar bastidor/markdown, promessa de fotos, e piso vs teto de orçamento interpretados pela LLM, promessa de trabalho futuro, e **anomalias ponta a ponta** (chat noturno negativo → alerta + restrição → `/api/kpis`; chat normal → sem alerta).

## Integração (subset executado junto das suítes)
- `apps/*/tests/integration/` sem falhas, incluindo `test_anomaly_pipeline.py` (contrato U5↔U1), `test_conversation_router.py` (consentimento explícito, eco de contato) e as rotas `/api/leads` do dashboard-api.

## Segurança (subset)
- `test_pii_masker.py`, `test_guardrails.py`, `test_pii_mask.py`, TTL, consentimento explícito (`test_conversation_polish.py`), `check_output_leak` — 0 falhas.
- Grep estático `eval/exec/os.system/subprocess` em código de produção: **1 ocorrência permitida** — `apps/voice-adapter/service/transcriber.py` (`ffmpeg` com lista de argumentos, sem `shell=True`, com timeout).

## Validação manual contra serviço real (não automatizada)
- **HubSpot via MCP (FR11.1), 2026-10-08:** `scripts/hubspot_authorize.py` autorizou o connector (OAuth 2.1 + PKCE) e listou 29 tools; `HubSpotCrmGateway` criou o contato de teste, achou o mesmo na 2ª chamada (sem duplicar) e atualizou o estágio. Em seguida um lead real do Telegram chegou ao HubSpot após o deploy.

## Performance
- NFR1.1/NFR1.2 continuam **Unverified** (alvo dominado por latência LLM/rede), owner: `performance-validation`.

## Evidência
- Saídas brutas ficaram no scratchpad da sessão (não versionadas). Reproduzir com os comandos de `build-instructions.md`.

## Loop-Back Log
(nenhuma entrada — nenhum command failure no run final. Achados corrigidos durante a rodada: falta de `import re` no handler; `dist-info` removido do pacote do crm-adapter; `budget`/`area`/`deadline` numéricos da LLM rejeitados pelo contrato do CRM (ADR-020).)
