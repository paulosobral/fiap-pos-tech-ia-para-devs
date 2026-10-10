# Test Results — Build and Test (run de 2026-10-08)

## Build status
- **compileall `apps`**: exit 0 — BUILD OK.
- `terraform validate` (infra/): Success. `bash -n start.sh stop.sh`: OK.
- Pacote do `crm-adapter` (com `mcp`): 27,7 MB compactado (< 50 MB).

## Resultados por unit (unitários + integração; sem o gate LLM)
| Unit | Passed | Failed | Skipped | Cobertura | Piso 80% |
|---|---|---|---|---|---|
| u1 conversation-router | 552 | 0 | 0 | 89,48% (633 linhas sem cobertura de 6016) | ✓ |
| u2 voice-adapter | 75 | 0 | 0 | 99,38% | ✓ |
| u3 crm-adapter | 141 | 0 | 0 | 99,12% | ✓ |
| u4 contact-ingest | 77 | 0 | 0 | 99,91% | ✓ |
| u5 anomaly-detector | 93 | 0 | 0 | 98,98% | ✓ |
| u6 followup | 93 | 0 | 0 | 98,52% | ✓ |
| u7 dashboard-api | 86 | 0 | 0 | 97,88% | ✓ |
| u7 dashboard-ui | 49 | 0 | 0 | 83,28% (perto do piso) | ✓ |

**Total: 1166 passed, 0 failed, 0 skipped** nas 8 aplicações, mais 4 do guarda de infra (`tests/infra`) = 1170

## Gate de qualidade com LLM real (`apps/conversation-router/tests/quality`)
- **59 passed, 0 failed** em 6 min 30 s (re-executado em 2026-10-10 após ADR-028: decisão de fechar, contato do lead, "última mensagem manda")
- Cobre: resolução de referências/ordinais, navegação, listas, fotos, contato/telefone (inclusive falado), fechamento de lead ponta a ponta, tipo de imóvel diferente do pedido sem narrar bastidor/markdown, promessa de fotos, e piso vs teto de orçamento interpretados pela LLM, promessa de trabalho futuro, e **anomalias ponta a ponta** (chat noturno negativo → alerta + restrição → `/api/kpis`; chat normal → sem alerta).

## Integração (subset executado junto das suítes)
- **Guarda de infra (ADR-024):** `tests/infra/test_dynamodb_indexes.py`, 4 testes, passando; sem a correção do `lead-index` em `sdr-alerts` ele reprova apontando as duas consultas quebradas (`anomaly-detector` e `restriction.py`). Contado à parte dos testes por aplicação.
- `apps/*/tests/integration/` sem falhas, incluindo `test_anomaly_pipeline.py` (contrato U5↔U1), `test_conversation_router.py` (consentimento explícito, eco de contato) e as rotas `/api/leads` do dashboard-api.

## Segurança (subset)
- **X-Ray sem segredos:** `test_tracing.py` (6 apps) proíbe `patch_all()`; com o SDK real, o token do bot aparece no trace com `patch_all` e **não** aparece com só `botocore` (verificação manual, daemon UDP simulado).
- `test_pii_masker.py`, `test_guardrails.py`, `test_pii_mask.py`, TTL, consentimento explícito (`test_conversation_polish.py`), `check_output_leak` — 0 falhas.
- Grep estático `eval/exec/os.system/subprocess` em código de produção: **1 ocorrência permitida** — `apps/voice-adapter/service/transcriber.py` (`ffmpeg` com lista de argumentos, sem `shell=True`, com timeout).

## Validação manual contra serviço real (não automatizada)
- **HubSpot via MCP (FR11.1), 2026-10-08:** `scripts/hubspot_authorize.py` autorizou o connector (OAuth 2.1 + PKCE) e listou 29 tools; `HubSpotCrmGateway` criou o contato de teste, achou o mesmo na 2ª chamada (sem duplicar) e atualizou o estágio. Em seguida um lead real do Telegram chegou ao HubSpot após o deploy.

## Pendência de verificação (2026-10-10)
- ADR-029: o gate completo com LLM real **não foi reexecutado** depois da mudança de fotos/IPTU (o responsável o roda no `start.sh`). Medições pontuais: 12/12 sem falsa promessa de fotos no fluxo completo; o teste `test_detail_missing_from_the_sheet_is_not_invented` oscila em ~3–10% por alucinação da LLM (aluguel tomado por IPTU) e a correção por prompt ainda não foi medida.

## Performance
- NFR1.1/NFR1.2 continuam **Unverified** (alvo dominado por latência LLM/rede), owner: `performance-validation`.

## Evidência
- Saídas brutas ficaram no scratchpad da sessão (não versionadas). Reproduzir com os comandos de `build-instructions.md`.

## Loop-Back Log
(nenhuma entrada — nenhum command failure no run final; X-Ray (ADR-021) adicionado depois, com a suíte reexecutada. Achados corrigidos durante a rodada: falta de `import re` no handler; `dist-info` removido do pacote do crm-adapter; `budget`/`area`/`deadline` numéricos da LLM rejeitados pelo contrato do CRM (ADR-020).)
