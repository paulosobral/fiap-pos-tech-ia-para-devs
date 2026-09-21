# Validation Report — Agente SDR Imobiliário (POC)

> Validação pré-provisionamento (design-time). A validação de runtime (pós `terraform apply`) é do `deployment-execution`; pós-deploy o `observability-setup` assume monitoração.

## Pré-condições de provisionamento

| # | Verificação | Método | Estado |
|---|---|---|---|
| V1 | Credenciais AWS válidas | `aws sts get-caller-identity` antes do `terraform plan` no `start.sh` (fase 0 a adicionar) | Script definido; execução no `deployment-execution` |
| V2 | Região configurada | `AWS_REGION`/`--region us-east-1` fixado em `provider "aws"` no `infra/providers.tf` | Definido (Q1) |
| V3 | Permissões mínimas | IAM do operador com admin do POC (hackathon); sem restrições de org | Aceito no escopo POC (anotado) |
| V4 | Quotas suficientes | Lambda concorrência padrão (1000) ≫ uso POC; Fargate 1×t3.micro < quota regional; DynamoDB on-demand sem quota prática | Conforme esperado |
| V5 | Secrets disponíveis | `TELEGRAM_BOT_TOKEN` e `INTERNAL_SECRET_TOKEN` criadas como SecureString; `PII_KMS_KEY_ID` aponta para a chave KMS criada no mesmo apply | Definido no inventário; ordem via `depends_on` no IaC |
| V6 | Conectividade de egress | Lambdas sem VPC → internet nato (Telegram API, CRM do grupo, provedor ASR) | Arquitetura valida (sem VPC, Q4) |

## Secrets & Parameter Store audit

| Segredo | Serviço | Injeção no runtime |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Secrets Manager SecureString | `aws_secretsmanager_secret_version` → env var da Lambda u1 (design u1: valores injetados pelo deploy) |
| `INTERNAL_SECRET_TOKEN` | Secrets Manager SecureString | env das Lambdas (rota `X-Internal-Secret` obrigatória — design u1) |
| `PII_KMS_KEY_ID` | KMS (chave própria) | env da Lambda; criptografia de PII (NFR5.x) |
| Credenciais CRM (upload do especialista) | env vars simples no POC (sem valor alto); migração p/ Secrets Manager fora do escopo | anotado como limite |

## Health checks definidos (executar pós-deploy no `deployment-execution`)

1. `curl -s "$API_URL/health"` → 200 (rota a implementar — cd-config fase 6)
2. Webhook Telegram: mensagem de teste → resposta em < 30s
3. Dashboard-ui: URL ECS abre e renderiza sessões
4. DLQs com 0 mensagens após o smoke test

## Achados

- **Nenhum bloqueante.** Notas: (a) `us-east-1` é proposta — confirmada na Q1; (b) backend Terraform local (sem lock) — single-operator assumido (cd-config tradeoff); (c) `faster-whisper`/ASR pesado no deploy real é risco conhecido (layer/snapshot), owner `deployment-execution`.
<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21) -->
