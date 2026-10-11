# Análise de Custo — Agente SDR Imobiliário (POC)

Sem faturamento real medido. A análise lista os vetores de custo e as escolhas que os mantêm baixos.

## Vetores de custo
| Vetor | Comportamento | Controle |
|---|---|---|
| LLM (OpenRouter) | maior custo variável: duas chamadas por turno, mais o gate de qualidade a cada `start.sh` (~6–7 min de chamadas reais) | tiering: DeepSeek (barato) primeiro, Haiku 4.5 e Sonnet 4.5 só em fallback |
| ECS Fargate (router, voice-adapter, dashboard) | custo por hora de task | tasks pequenas (router com 256 CPU/512 MB) e janela 09:00–18:00 BRT; fora dela, 0 tasks |
| Lambdas, API Gateway, SQS, DynamoDB | pago por uso, baixo no POC | sob demanda |
| Secrets Manager, KMS | custo fixo por segredo/chave | poucos segredos; ambiente destruído no `stop.sh` |
| Evitado | NAT Gateway (~US$32/mês) e ALB (~US$16/mês) | sem VPC custom, IP público efêmero |

## Recomendações
1. Rodar o `stop.sh` ao fim de cada sessão (o ambiente não deve ficar de pé).
2. Registrar o custo de LLM por conversa (tokens já vêm na resposta do provedor); métrica `CostMonthly` nunca foi emitida.
3. Reavaliar o gate com LLM real se o crédito do OpenRouter ficar apertado (hoje roda a cada subida).
