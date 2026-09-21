# Environment Provisioning Questions

> Respostas derivadas do contexto afirmado (`team.md` § Deployment: deploy manual via `start.sh`/`stop.sh`; gate do `ci-pipeline`: módulos HashiCorp, dist.zip por Lambda; design u1: segredos via Secrets Manager; escopo feature/POC — custo domina).

## Q1 — Região AWS
- A. **`us-east-1`** (Recommended) — menor latência global (API GW/CloudFront edge), disponibilidade total dos serviços usados, preço padrão
- B. `sa-east-1` (São Paulo) — mais próximo do Brasil (+30–50% de preço, menor latência p/ Telegram BR)

[Answer]: A

## Q2 — Credenciais AWS
- A. **Perfil/env vars do operador via AWS CLI** (Recommended) — `aws sts get-caller-identity` valida no `start.sh` (V1); zero setup extra
- B. Usuário IAM dedicado no `infra/` — bootstrap circular (precisa de credenciais p/ criar as credenciais)

[Answer]: A

## Q3 — Segredos e criptografia
- A. **Secrets Manager (`TELEGRAM_BOT_TOKEN`, `INTERNAL_SECRET_TOKEN`) + chave KMS própria p/ PII** (Recommended) — conforme design u1 (`PII_KMS_KEY_ID`, NFR5.x); `depends_on` resolve a ordem
- B. SSM Parameter Store (free tier) — economiza ~US$0,40/secreto/mês, mas o design já fixou Secrets Manager e o custo POC é mínimo

[Answer]: A

## Q4 — Redes (VPC/SG/NACL)
- A. **Sem VPC custom: Lambdas fora de VPC, ECS na VPC default** (Recommended) — egress nato p/ Telegram/CRM/ASR; zero custo de NAT Gateway; DynamoDB/SQS gerenciados
- B. VPC custom com SG por serviço — custo de NAT (~US$32/mês) e complexidade IaC sem benefício no POC

[Answer]: A

## Consolidated Summary Confirmation

<!-- preenchido antes do approval gate -->

[Answer]: Looks correct