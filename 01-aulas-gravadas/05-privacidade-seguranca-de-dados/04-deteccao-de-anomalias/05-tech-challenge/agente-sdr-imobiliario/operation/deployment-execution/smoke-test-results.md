# Smoke Test Results — deployment-execution

**Data:** 2026-09-21 · **Script:** start.sh fase [6/6] · **API:** `https://izlmelzop8.execute-api.us-east-1.amazonaws.com`

| # | Check | Esperado | Resultado |
|---|-------|----------|-----------|
| 1 | `GET /health` | 200 `{"status":"ok"}` | **OK (200)** — rota adicionada ao conversation-router (u1) |
| 2 | `POST /webhook/telegram` sem `X-Telegram-Bot-Api-Secret-Token` | 400 ou 401 | **OK (401)** — rejeição correta |
| 3 | `GET /api/kpis` com `Authorization: Bearer poc-smoke` | 200 | **OK (200)** — handler exige presença do bearer (POC) |

Verificação manual pós-pipeline (curl): `/health` 200, `/api/kpis` com header 200 — consistente.

## Observações

- O POC **não valida o valor** do Bearer em /api/kpis (presença apenas). Evolução pós-POC: validar contra `sdr/sdr-internal-secret-token` (Secrets Manager) ou Cognito.
- O dashboard (ECS) não é coberto pelo smoke do pipeline — roda com desired_count fora da janela (09:00–17:00 BRT) e IP público efêmero. Validação manual na janela: Console ECS > cluster `sdr` > service > task > IP público :80.
- O webhook de produção com Telegram real exige o token em Secrets Manager + re-apply (ver health-check-report).
