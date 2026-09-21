# Rollback Runbook — Agente SDR Imobiliário (POC)

> Consumes: `deployment-strategy.md`, `cd-config.md`, `construction/ci-pipeline/ci-config.md` (`stop.sh` = `terraform destroy`, NFR9.3). Execução é do `deployment-execution`.

## Quando acionar

| Gatilho | Rollback aplicável |
|---|---|
| `terraform apply` falha no meio | R1 (reconciliar) |
| Comportamento errado após deploy de código | R2 (reverter código) |
| Mudança de infra destruiu dados/estágio | R3 (reverter infra via git) |
| Ambiente instável / custo fora de controle (ex.: modelo pesado de ASR rodando) | R4 (teardown completo) |

## R1 — `apply` falhou no meio

1. Re-ler a mensagem de erro do `terraform apply` (não re-ran cegamente).
2. `cd infra/ && terraform plan` — estado atual vs. código.
3. Corrigir a causa (código `.tf` ou credenciais AWS) e re-rodar `start.sh` — idempotente por design.

## R2 — Reverter código de uma aplicação

1. `deployment-execution` retém versões em `dist/archive/<app>-<YYYYMMDDHHmm>.zip` a cada deploy bem-sucedido (contrato para o `start.sh`).
2. Restaurar o zip anterior: `cp dist/archive/<app>-<ts>.zip dist/<app>.zip`.
3. `cd infra/ && terraform apply -target=module.<app>` — atualiza só essa Lambda (zip novo → `source_code_hash` muda).
4. Smoke check (cd-config.md fase 6): `curl $API_URL/health` + suíte mínima pytest.

## R3 — Reverter mudança de infra

1. `git revert` do commit que alterou `infra/*.tf` (branch feature — histórico linear por PR).
2. Se o revert apagar recurso com dados (DynamoDB/SQS), **parar**: `terraform plan` deve mostrar `destroy` na tabela — recusar e considerar R4 + exportar dados antes (`aws dynamodb scan`).

## R4 — Teardown completo e reinício limpo

1. `./stop.sh` — `terraform destroy` (NFR9.3): derruba Lambdas, API, tabelas, filas, ECS.
2. Corrigir a causa-raiz (código/config).
3. `./start.sh` — reconstrói do zero (build → gates → apply).
4. Validação pós-restart: smoke checks do cd-config + testes de integração rodam antes do apply (fase 3 do start.sh).

## Validação pós-rollback (obrigatória)

- [ ] `curl -s "$API_URL/health"` responde 200
- [ ] Webhook do Telegram responde (enviar mensagem de teste no bot)
- [ ] Suíte pytest mínima verde (conversation-router — caminho crítico)
- [ ] Dashboard abre e mostra dados de sessão
- [ ] DLQ com 0 mensagens (ou investigar em `observability-setup`)

<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21, authorization d8572d71) -->