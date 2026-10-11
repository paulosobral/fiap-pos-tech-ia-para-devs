# Alarmes — Agente SDR Imobiliário (POC)

## Estado atual
Nenhum alarme do CloudWatch nem tópico SNS está provisionado (`infra/` não tem `aws_cloudwatch_metric_alarm`). A detecção de problemas é manual: logs, X-Ray, DLQs e o dashboard de negócio.

## Alarmes propostos para a evolução (abaixo do SLO, para dar tempo de agir)
| Alarme | Sinal | Limite sugerido |
|---|---|---|
| DLQ com mensagens | `ApproximateNumberOfMessagesVisible` em `sdr-voice-dlq` e demais DLQs | > 0 por 5 min |
| Erros de Lambda | `Errors` por função | ≥ 3 em 5 min |
| Router fora do ar | tasks ECS em execução do `conversation-router` | 0 dentro da janela 09:00–18:00 BRT |
| Latência do turno | p90 do tempo de resposta do router | > 12 s por 15 min |
| Falhas de LLM | contagem de `429`/timeout nos logs `roteador:` | > 20% das chamadas em 15 min |

Notificação: tópico SNS com e-mail do responsável (o SES já está provisionado para o follow-up).

## Verificação manual enquanto não houver alarmes
- Fila `sdr-*-dlq` vazia (checklist do `rollback-runbook.md`).
- `GET /api/kpis` respondendo (smoke da fase 6 do `start.sh`).
- Erros no X-Ray (as exceções do `lead-index` foram achadas assim).
