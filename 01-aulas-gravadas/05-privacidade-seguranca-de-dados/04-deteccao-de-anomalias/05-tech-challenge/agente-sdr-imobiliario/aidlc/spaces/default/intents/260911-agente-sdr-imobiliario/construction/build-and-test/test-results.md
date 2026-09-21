# Test Results — Build and Test (run de 2026-09-21)

## Build status
- **compileall `apps`**: exit 0 — BUILD OK.
- Smoke de import: realizado implicitamente pelas suítes (conftest insere `apps/<app>` no `sys.path`; handlers importados durante os testes). Nenhuma falha.

## Resultados por unit (comandos das instruction files, rodados 1× cada; `--cov-fail-under=80`)
| Unit | Comando (resumido) | Exit | Passed | Failed | Skipped | Cobertura | Piso 80% |
|---|---|---|---|---|---|---|---|
| u1 conversation-router | `pytest apps/conversation-router/tests --cov=apps/conversation-router --cov-fail-under=80 -q` (dedupe do comando longo da instruction) | 0 | 116 | 0 | 0 | 96.51% | ✓ |
| u2 voice-adapter | `pytest apps/voice-adapter/tests --cov=apps/voice-adapter --cov-fail-under=80 -q` | 0 | 75 | 0 | 0 | 99.88% | ✓ |
| u3 crm-adapter | `pytest apps/crm-adapter/tests --cov=apps/crm-adapter --cov-fail-under=80 -q` | 0 | 117 | 0 | 0 | 99.59% | ✓ |
| u4 contact-ingest | `pytest apps/contact-ingest/tests --cov=apps/contact-ingest --cov-fail-under=80 -q` | 0 | 71 | 0 | 0 | 100.00% | ✓ |
| u5 anomaly-detector | `pytest apps/anomaly-detector/tests --cov=apps/anomaly-detector --cov-fail-under=80 -q` | 0 | 84 | 0 | 3 | 95.29% | ✓ |
| u6 followup | `pytest apps/followup/tests --cov=apps/followup --cov-fail-under=80 -q` | 0 | 87 | 0 | 0 | 98.52% | ✓ |
| u7 dashboard-api | `pytest apps/dashboard-api/tests --cov=apps/dashboard-api --cov-fail-under=80 -q` | 0 | 66 | 0 | 0 | 98.10% | ✓ |
| u7 dashboard-ui | `pytest apps/dashboard-ui/tests -q` (smoke) | 0 | 8 | 0 | 2 | n/a (smoke) | ✓ |

**Total: 624 passed, 0 failed, 5 skipped** (5 skipped: 3 em u5 — features dependentes de ambiente; 2 em dashboard-ui — smoke).

## Integração (subset executado junto das suítes)
- `apps/*/tests/integration/` executados como parte dos comandos acima (0 falhas), incluindo `test_anomaly_pipeline.py` (contrato U5↔U1) e `test_conversation_router.py` (fluxo ponta-a-ponta).

## Segurança (subset)
- `test_pii_masker.py`, `test_guardrails.py`, `test_pii_mask.py`, stores com TTL, consent — todos dentro dos comandos acima (0 falhas).
- Grep estático `eval/exec/os.system/subprocess` em `apps` (não-testes): **CLEAN** (nenhuma ocorrência).

## Performance
- Execução local: não executável com fidelidade (alvo dominado por latência LLM/rede). **NFR1.1/NFR1.2 → Unverified, owner: `performance-validation`** (stage agendado no plano). Benchmark local de lógica: não mandatado (op de guarda disponível em `performance-test-instructions.md`).

## Log de evidência
- Saídas brutas por suíte: `/tmp/opencode/bt-u{1..6}.log`, `bt-u7-api.log`, `bt-u7-ui.log`; resumo: `/tmp/opencode/bt-results.txt`.

## Loop-Back Log
(nenhuma entrada — nenhum command failure ou fix in-stage nesta execução)
