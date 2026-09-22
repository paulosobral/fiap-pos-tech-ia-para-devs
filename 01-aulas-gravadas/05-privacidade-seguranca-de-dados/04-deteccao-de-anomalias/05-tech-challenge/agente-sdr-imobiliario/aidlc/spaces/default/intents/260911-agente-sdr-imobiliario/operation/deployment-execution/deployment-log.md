# Deployment Log — agente SDR imobiliário (POC)

**Estágio:** deployment-execution · **Data:** 2026-09-21 · **Executor:** start.sh (raiz)

## 1. Pipeline executado (start.sh)

| Fase | Conteúdo | Resultado |
|------|----------|-----------|
| [1/6] Setup | venv Python 3.11 + deps dev | OK |
| [2/6] Compile | `compileall` 8 apps | OK |
| [3/6] Testes | pytest ×8 gates (cov ≥ 80% + UI smoke) | OK (624 tests) |
| [4/6] Package | dist zips ×7 Lambdas + archive | OK (7 artefatos) |
| [5/6] Deploy | terraform init → apply1 (base+ECR) → podman build/push → apply2 (imagem ECR) | OK |
| [5b/6] Imagem | `podman build` + push `sdr-dashboard-ui:poc-20260921185026` | OK |
| [6/6] Smoke | GET /health · webhook sem secret · GET /api/kpis | OK (3/3) |

**Resultado final:** `SMOKE OK` · API `https://izlmelzop8.execute-api.us-east-1.amazonaws.com` · EXIT=0.

## 2. Iterações e correções durante o deploy

1. **Reempacotamento pelo módulo Lambda** (`poetry_install_step` UnicodeDecodeError no pyproject) → substituído `source_path` por `create_package = false` + `local_existing_package` nos 7 módulos (nome com hífen, não underline).
2. **SecretString vazio** (Secrets Manager InvalidRequest) → `secret_version` do Telegram token condicional (`count = var.telegram_bot_token == "" ? 0 : 1`); Lambdas leem o token de `var.telegram_bot_token`.
3. **KMS MalformedPolicyDocumentException** ("will not allow you to update the key policy") → policy com statement root admin (`arn:aws:iam::<account>:root`, `kms:*`) + uso da Lambda.
4. **Voice zip > limite Lambda 50MB (413)** → zip foi reconstruído sobre o antigo (zip não apaga entradas prévias): `rm -f` do zip antes de regerar; POC empacota voice-adapter só com `requests` (boto3 nativo do runtime) — faster-whisper/ffmpeg/ctranslate2 (~120MB) NÃO cabem no pacote direto; transcrição real fica para pós-POC (layer/Amazon Transcribe).
5. **Event source mapping "role does not have permissions to call ReceiveMessage"** → módulo `terraform-aws-modules/lambda/aws ~> 7.0`: `attach_policies = true` exige `number_of_policies` (default 0 não attacha). Adicionado `number_of_policies = 1` nos 7 módulos.
6. **ECS RegisterTaskDefinition "Container.image should not be null or empty"** → default de `dashboard_ui_image` passa a ser imagem pública `public.ecr.aws/docker/library/python:3.11-slim` (permite o apply1 antes do push ECR); start.sh substitui no apply2.
7. **Smoke 401 em /api/kpis** → handler do dashboard-api exige presença de `Authorization: Bearer` (POC não valida valor); smoke atualizado para enviar o header.

## 3. Recursos provisionados (conta 144842881551, us-east-1)

- HTTP API `sdr-http-api` — rotas POST /webhook/telegram, POST /internal/{proxy+}, GET /health, GET /api/{proxy+} ($default auto_deploy).
- 7 Lambdas `sdr-*` (packages zip em dist/; voice-adapter degradado — ver health-check-report).
- DynamoDB ×5 (sdr-sessions c/ GSI lead-index + TTL, sdr-pii, sdr-alerts, sdr-ingest-dedupe, sdr-followup-state).
- SQS ×6 (voice/crm/ingest queues + 3 DLQs maxReceiveCount 3) + 3 event source mappings (ids 0f153c99…, a7f5e9b7…, 8bd25fd1…).
- EventBridge ×2 (anomaly rate(1 minute), followup rate(1 day)).
- KMS key `pii` + alias; Secrets Manager `sdr/tg-bot-token` (vazia até o humano preencher) e `sdr/dashboard-api-token` (random_password 32).
- ECR `sdr-dashboard-ui` + imagem `poc-20260921185026`; ECS cluster `sdr` + task def 256/512 + service desired_count 0; autoscaling scheduled 09:00–17:00 BRT (0→1→0); SG ingress 80; log group 7d.
- State terraform local (`infra/terraform.tfstate`) — backend local (pós-POC: S3).

## 4. Teardown

`./stop.sh` (terraform destroy; `AUTO=1` para não interativo).
