# Plano de Resposta a Incidentes — Agente SDR Imobiliário (POC)

## Escopo
POC acadêmico com ambiente efêmero (`start.sh`/`stop.sh`), dados de lead reais só em demonstração e um único responsável técnico. Não há plantão nem AWS Incident Manager.

## Classificação
| Nível | Exemplo | Resposta |
|---|---|---|
| P1 | vazamento de segredo (token do Telegram, chave de LLM, token do HubSpot) ou de dado pessoal | conter primeiro: rotacionar o segredo, rodar `stop.sh` se preciso; registrar o fato |
| P2 | bot fora do ar durante demonstração; leads não chegam ao CRM | seguir RB1 ou RB3 |
| P3 | lentidão, alerta falso, gráfico vazio | tratar no próximo ciclo |

## Fluxo
1. **Detectar** (logs, X-Ray, DLQ, dashboard; não há alarme automático).
2. **Conter** (parar o serviço afetado ou `stop.sh`).
3. **Corrigir** com o runbook correspondente.
4. **Validar** com o checklist pós-rollback do `rollback-runbook.md`.
5. **Registrar**: para P1/P2, uma nota curta com linha do tempo, causa raiz e prevenção, anexada ao diário do estágio.

## Recuperação (RTO/RPO)
- **RTO**: o tempo de um `start.sh` completo (build, gates e apply); reconstrução do zero.
- **RPO**: as sessões de conversa e os alertas ficam no DynamoDB do ambiente efêmero e **se perdem** no `stop.sh` (não há AWS Backup nem PITR). Os leads já enviados ficam no HubSpot, que é a fonte de verdade comercial.

## Dados pessoais
Em incidente com dado pessoal (LGPD), o responsável avalia a comunicação ao titular e à ANPD. O POC guarda PII criptografada com KMS em `sdr-pii` e não grava dados em logs.
