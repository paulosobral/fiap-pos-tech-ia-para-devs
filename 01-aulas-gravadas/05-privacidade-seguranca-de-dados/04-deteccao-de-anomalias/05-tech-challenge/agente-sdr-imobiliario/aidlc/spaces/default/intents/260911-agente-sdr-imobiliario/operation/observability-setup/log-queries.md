# Consultas de Log — Agente SDR Imobiliário (POC)

Logs do router e dos serviços ECS ficam em `/ecs/…` (retenção 7 dias); os das Lambdas em `/aws/lambda/…` (14 dias). O arquivo `logs/start-latest.log` guarda a saída do `start.sh` e `logs/stop-latest.log` a do `stop.sh`.

## Logs Insights (exemplos)
```
# decisões do roteador por mensagem
fields @timestamp, @message
| filter @message like /roteador:/
| sort @timestamp desc
| limit 50
```
```
# erros do router
fields @timestamp, @message
| filter @message like /ERROR|Traceback|Exception/
| sort @timestamp desc
| limit 50
```
```
# limite de taxa ou falha do provedor de LLM
fields @timestamp, @message
| filter @message like /429|RateLimit|fallback/
| stats count() by bin(15m)
```
```
# envio ao CRM (HubSpot)
fields @timestamp, @message
| filter @message like /hubspot|crm/
| sort @timestamp desc
| limit 50
```

## Regras de higiene
Os logs não devem conter o token do Telegram nem dados pessoais em claro (PII fica no DynamoDB `sdr-pii`, criptografado com KMS). O SDK do X-Ray é aplicado só ao `botocore` justamente para não gravar a URL do Telegram, que leva o token.
