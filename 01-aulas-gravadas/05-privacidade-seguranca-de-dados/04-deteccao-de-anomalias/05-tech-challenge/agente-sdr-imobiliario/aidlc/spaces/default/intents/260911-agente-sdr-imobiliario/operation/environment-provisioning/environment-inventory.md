# Environment Inventory — Agente SDR Imobiliário (POC)

> Consumes: `operation/deployment-pipeline/cd-config.md` (pipeline local, módulos HashiCorp), NFR9.1–9.4 (IaC por serviço, DLQ, autoscaling), NFR5.x (segredos/LGPD), design u1–u7 (`TELEGRAM_BOT_TOKEN`, `PII_KMS_KEY_ID`, `INTERNAL_SECRET_TOKEN` via Secrets Manager).

## Ambiente-alvo

| Atributo | Valor | Origem |
|---|---|---|
| Conta AWS | Conta do hackathon (credenciais do humano via AWS CLI perfil/env vars) | Q2 do questions |
| Região | `us-east-1` (proposta; baixa latência global API GW + disponibilidade de serviços) | Q1 do questions |
| Provisionador | `terraform apply` via `start.sh` (backend local) | cd-config.md |
| Teardown | `terraform destroy` via `stop.sh` | NFR9.3 |

## Inventário de recursos alvo (materializados pelo `deployment-execution`)

| Recurso | Quantidade/Spec | Módulo | Owner |
|---|---|---|---|
| Lambdas python3.11 (u1–u6, dashboard-api) | 8 × `dist/<app>.zip`, `handler.handler` | `terraform-aws-modules/lambda/aws` | deployment-execution |
| HTTP API + rotas webhook/internals | 1 API (endpoint estável) | `terraform-aws-modules/apigateway-v2/aws` | deployment-execution |
| DynamoDB (sessões, leads, alertas; TTL NFR3.3, GSI lead-index) | 3 tables on-demand | `terraform-aws-modules/dynamodb-table/aws` | deployment-execution |
| SQS filas async + DLQ | filas u2/u3/u4 + 3 DLQs (NFR4.1) | `terraform-aws-modules/sqs/aws` | deployment-execution |
| EventBridge scheduler | regra de anomalias/u6 (NFR8.1) | `aws_eventbridge_*` nativo | deployment-execution |
| ECS Fargate (dashboard-ui Streamlit) | 1 task `t3.micro`, subnet pública da VPC default | `terraform-aws-modules/ecs/aws` | deployment-execution |
| Secrets Manager | `TELEGRAM_BOT_TOKEN`, `INTERNAL_SECRET_TOKEN` (SecureString) | `aws_secretsmanager_*` nativo | deployment-execution |
| KMS (PII, `PII_KMS_KEY_ID`) | 1 chave simétrica | `aws_kms_*` nativo | deployment-execution |

## Redes

- **Sem VPC custom no POC** (Q4 do questions file): Lambdas executam fora de VPC (acesso nato à internet p/ Telegram/CRM/ASR); DynamoDB/SQS/API GW são gerenciados; ECS Fargate usa a **VPC default** (subnets públicas). NACL/SG custom ficam fora do escopo — SG mínimo no ECS.

## O que este estágio NÃO provisiona

- Este estágio é **validação de design/pré-condições** — o provisionamento real (`terraform apply`) é executado pelo **`deployment-execution`** (decisão registrada no gate do ci-pipeline). Nenhum `terraform apply` é rodado aqui.