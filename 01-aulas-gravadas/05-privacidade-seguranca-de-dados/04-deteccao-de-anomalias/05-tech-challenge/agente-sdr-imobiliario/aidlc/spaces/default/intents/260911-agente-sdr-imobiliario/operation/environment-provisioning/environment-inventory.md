# Environment Inventory — Agente SDR Imobiliário (POC)

> Consumes: `operation/deployment-pipeline/cd-config.md` (pipeline local, módulos HashiCorp), NFR9.1–9.4 (IaC por serviço, DLQ, autoscaling), NFR5.x (segredos/LGPD), design u1–u7 (`TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`, `PII_KMS_KEY_ID`, `INTERNAL_SECRET_TOKEN` via Secrets Manager).

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
| Secrets Manager | `TELEGRAM_BOT_TOKEN`, `LLM_API_KEY`, `INTERNAL_SECRET_TOKEN` (SecureString) | `aws_secretsmanager_*` nativo | deployment-execution |
| KMS (PII, `PII_KMS_KEY_ID`) | 1 chave simétrica | `aws_kms_*` nativo | deployment-execution |

## Redes

- **Sem VPC custom no POC** (Q4 do questions file): Lambdas executam fora de VPC (acesso nato à internet p/ Telegram/CRM/ASR); DynamoDB/SQS/API GW são gerenciados; ECS Fargate usa a **VPC default** (subnets públicas). NACL/SG custom ficam fora do escopo — SG mínimo no ECS.

## O que este estágio NÃO provisiona

- Este estágio é **validação de design/pré-condições** — o provisionamento real (`terraform apply`) é executado pelo **`deployment-execution`** (decisão registrada no gate do ci-pipeline). Nenhum `terraform apply` é rodado aqui.
<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21) -->

## Atualização (2026-10-09) — X-Ray (ADR-021)

Recursos acrescentados ao IaC depois da validação inicial; o apply real continua sendo do `deployment-execution`.

| Recurso | Quantidade/Spec | Onde | Owner |
|---|---|---|---|
| Tracing ativo nas Lambdas | `tracing_mode = "Active"` + `attach_tracing_policy = true` nos 7 módulos | `infra/lambda-*.tf` | deployment-execution |
| Contêiner auxiliar `xray-daemon` | 1 por task do `conversation-router`; imagem pública `public.ecr.aws/xray/aws-xray-daemon:3.7.0` (variável `xray_daemon_image`), UDP 2000, `essential = false`, 32 CPU / 64 MB reservados, log no mesmo log group do router | `infra/ecs.tf` | deployment-execution |
| Variáveis do SDK na task do router | `AWS_XRAY_DAEMON_ADDRESS=127.0.0.1:2000`, `AWS_XRAY_CONTEXT_MISSING=LOG_ERROR` | `infra/ecs.tf` | deployment-execution |
| Permissões do X-Ray | declaração `XRay` na política `sdr_lambda` (`PutTraceSegments`, `PutTelemetryRecords`, `GetSampling*`), reaproveitada pela role da task do router | `infra/iam.tf` | deployment-execution |

Limites: o API Gateway HTTP (v2) não suporta X-Ray; `voice-adapter` e `dashboard-ui` (ECS) não têm tracing.

## Atualização (2026-10-09) — nome da assistente (ADR-022)

| Recurso | Quantidade/Spec | Onde | Owner |
|---|---|---|---|
| Parâmetro SSM `/sdr/bot-name` | 1 `String` (não sensível), valor da variável `bot_name` (padrão `Cecília`), `overwrite = true` | `infra/secrets.tf` | deployment-execution |
| Variável `BOT_NAME_SSM` na task do router | aponta para o parâmetro acima; a role já lê `/sdr/*` | `infra/ecs.tf` | deployment-execution |

## Atualização (2026-10-10) — índice de alertas (ADR-024)

| Recurso | Quantidade/Spec | Onde | Owner |
|---|---|---|---|
| GSI `lead-index` em `sdr-alerts` | `hash_key = lead_id` (S), projeção `ALL`; criado online no `apply`, sem recriar a tabela | `infra/dynamodb.tf` | deployment-execution |
