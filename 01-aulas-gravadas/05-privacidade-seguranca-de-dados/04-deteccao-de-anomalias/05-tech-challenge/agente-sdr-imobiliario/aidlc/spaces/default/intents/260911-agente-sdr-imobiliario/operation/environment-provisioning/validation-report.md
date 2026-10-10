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
| V7 | X-Ray: permissões e imagem do daemon | declaração `XRay` na política `sdr_lambda`; imagem pública do daemon puxável pela task (IP público, sem NAT) | Definido no IaC (ADR-021); confirmação no `deployment-execution` |

## Secrets & Parameter Store audit

| Segredo | Serviço | Injeção no runtime |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | Secrets Manager SecureString | `aws_secretsmanager_secret_version` → env var da Lambda u1 (design u1: valores injetados pelo deploy) |
| `INTERNAL_SECRET_TOKEN` | Secrets Manager SecureString | env das Lambdas (rota `X-Internal-Secret` obrigatória — design u1) |
| `PII_KMS_KEY_ID` | KMS (chave própria) | env da Lambda; criptografia de PII (NFR5.x) |
| Token do bot do Telegram nos traços do X-Ray | **não** vai: só o `botocore` é instrumentado; `patch_all()` gravaria a URL com o token (ADR-021) | regressão coberta por `test_tracing.py` |
| `/sdr/bot-name` | SSM Parameter Store `String` (nome da assistente, não é segredo) | `BOT_NAME_SSM` na task do router; cache de 5 min |
| Credenciais CRM (upload do especialista) | env vars simples no POC (sem valor alto); migração p/ Secrets Manager fora do escopo | anotado como limite |

## Health checks definidos (executar pós-deploy no `deployment-execution`)

1. `curl -s "$API_URL/health"` → 200 (rota a implementar — cd-config fase 6)
2. Webhook Telegram: mensagem de teste → resposta em < 30s
3. Dashboard-ui: URL ECS abre e renderiza sessões
4. DLQs com 0 mensagens após o smoke test
5. **X-Ray (ADR-021, ainda não validado numa subida real):** depois de uma conversa, Console X-Ray → Traces mostra o segmento `conversation-router` com subsegmentos `llm:<modelo>` e as Lambdas no service map; o log do contêiner `xray-daemon` não tem erro de permissão ou região
6. **Nome da assistente (ADR-022):** `aws ssm get-parameter --name /sdr/bot-name` devolve `Cecília`; a primeira mensagem do bot no Telegram começa com "Olá! Meu nome é Cecília"; trocar o valor com `put-parameter` vale em até 5 minutos, sem deploy
7. **Alertas (ADR-024):** `aws dynamodb describe-table --table-name sdr-alerts` lista o GSI `lead-index` com `IndexStatus: ACTIVE`; o log da Lambda `sdr-anomaly-detector` não tem `alert_store_unavailable`, e o X-Ray não mostra `ValidationException`
8. **Decisão do roteador no log (ADR-025):** depois de uma conversa, o log do `conversation-router` tem uma linha `roteador: tool=... args=... pensamento=...` por mensagem; é por ela que se explica uma escolha de ferramenta inesperada
11. **Encerramento (ADR-031):** depois de o lead passar o WhatsApp/e-mail, a resposta termina com "Seu atendimento está encerrado…"; a mensagem seguinte recebe a apresentação e o consentimento de novo, e o dashboard mostra DOIS leads para a mesma pessoa (o encerrado e o novo)
10. **Primeira conversa (ADR-026):** depois de "oi" e "sim", a resposta pergunta compra, locação ou investimento e não lista imóveis; a primeira mensagem diz que o nome vem do perfil do Telegram e que só se pede WhatsApp ou e-mail
9. **X-Ray sem segredos:** abrir um trace do `followup`/router e confirmar que nenhuma URL com `/bot<TOKEN>/` aparece

## Achados

- **X-Ray (2026-10-09):** o sidecar do daemon (`-o -n <região>`) no Fargate e o tracing nas Lambdas estão implementados e testados com SDK real/daemon simulado, mas só se confirmam num deploy (health checks 5 e 6).
- **Nenhum bloqueante.** Notas: (a) `us-east-1` é proposta — confirmada na Q1; (b) backend Terraform local (sem lock) — single-operator assumido (cd-config tradeoff); (c) `faster-whisper`/ASR pesado no deploy real é risco conhecido (layer/snapshot), owner `deployment-execution`.
<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21) -->
