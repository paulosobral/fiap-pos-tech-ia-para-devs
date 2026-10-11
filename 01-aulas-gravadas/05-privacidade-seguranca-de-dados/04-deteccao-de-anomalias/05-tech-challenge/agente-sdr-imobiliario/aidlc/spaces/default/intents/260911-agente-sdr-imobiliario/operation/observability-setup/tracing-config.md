# Rastreamento Distribuído (X-Ray) — Agente SDR Imobiliário (POC)

## Configuração
- **Lambdas**: `tracing_config { mode = "Active" }`; o SDK `aws-xray-sdk` instrumenta chamadas ao `botocore` (DynamoDB, SQS, Secrets Manager etc.).
- **Router (ECS)**: o daemon do X-Ray roda como contêiner auxiliar na mesma task; o SDK também instrumenta só o `botocore`.
- **API Gateway HTTP v2**: não oferece X-Ray; o traço começa no router/Lambda.
- **Permissões**: `xray:PutTraceSegments`, `PutTelemetryRecords` e os `GetSampling*` nos papéis (`infra/iam.tf`).
- **Testes**: `tests/unit/test_tracing.py` de cada aplicação; há teste que proíbe `patch_all()`.

## Decisão de segurança
`patch_all()` fica proibido: ele instrumentaria `requests`/`httpx` e gravaria nos traços a URL da API do Telegram, que contém o token do bot.

## Uso
No console do X-Ray: mapa de serviços e busca por traços com erro (foi assim que apareceu o `ValidationException` do índice `lead-index` em `sdr-alerts`) e por latência alta (para decompor os 8–15 s por turno em LLM, DynamoDB e CRM). Esta configuração ainda não foi confirmada numa subida real depois da última mudança.
