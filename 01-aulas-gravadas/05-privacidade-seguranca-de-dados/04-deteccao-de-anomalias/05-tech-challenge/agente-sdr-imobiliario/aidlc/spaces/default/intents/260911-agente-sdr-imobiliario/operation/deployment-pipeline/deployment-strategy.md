# Deployment Strategy — Agente SDR Imobiliário (POC)

> Consumes: `cd-config.md` (pipeline local), `construction/ci-pipeline/ci-config.md` (módulos HashiCorp, dist.zip por Lambda), NFR8.1 (autoscaling), NFR9.2 (IaC por serviço).

## Estratégia escolhida

**In-place update via Terraform** — `terraform apply` sobre o ambiente efêmero único. Blue/green, canary e rolling entre réplicas ficam **fora do escopo POC** (ambiente manual, sem produção formal; custo domina).

| Componente | Update strategy | Detalhe |
|---|---|---|
| 7 Lambdas (u1–u6, dashboard-api) | `terraform apply` com novo `dist/<app>.zip` | módulo `terraform-aws-modules/lambda/aws` faz bump de versão/`source_code_hash`; alias `$LATEST` na rota do API Gateway |
| HTTP API (apigateway-v2) | in-place | endpoint e URL **estáveis** (mesmo `terraform apply` não recria a API sem mudança de props) — webhook do Telegram não quebra a cada deploy |
| DynamoDB (sessões, leads, alertas) | in-place, dados preservados | mudança de schema = `terraform plan` mostra `in-place attribute update`; destructive change exige decisão explícita (ver rollback-runbook) |
| SQS + DLQ | in-place | criação de fila nova não derruba a antiga |
| dashboard-ui (Streamlit/ECS Fargate) | rolling de 1 task | novo `dist/dashboard-ui.zip` → nova task definition → substituição da task (janela < 1 min) |

## Zero-downtime?

- **Não garantido no POC** — janela de manutenção aceita (deploy manual, horário escolhido pelo humano). O que preserva: endpoint do API Gateway estável, dados DynamoDB preservados em updates in-place, DLQ não perde mensagens em redeploy de consumidores.
- Lambda + API GW escalam por natureza (NFR8.1) — não há réplicas a balancear; "rolling" só existe no ECS (1 task).

## Idempotência e reexecução

- `start.sh` é **idempotente**: re-run após falha re-aplica `terraform apply` (estado consistente) e re-zipa `dist/`.
- `terraform plan` antes de cada `apply` mostra o diff; humano revisa (gate de aprovação).

<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21, authorization d8572d71) -->