# Observability Setup Questions

> Respostas derivadas do que o projeto já tem (`infra/`, `apps/*/tracing.py`, `build-and-test`, `deployment-execution`) e da decisão de manter o POC enxuto.

## Q1 — Sinais de ouro
- A. **Latência do turno, erros, tráfego de mensagens e saturação das filas/DLQ**, observados por logs, X-Ray e dashboard de negócio (Recommended)
- B. Painel completo de métricas customizadas desde já — exige emitir métricas que o código não emite

[Answer]: A

## Q2 — SLOs
- A. **Metas da POC medidas manualmente** (p90 < 10 s, 99% de disponibilidade na janela, 0 mensagens em DLQ, 95% dos leads no CRM); medição automática fica para depois (Recommended)
- B. SLOs com orçamento de erro e alarmes automáticos agora

[Answer]: A

## Q3 — Dashboards
- A. **Só o dashboard de negócio (Streamlit)** mais logs e X-Ray; sem dashboard CloudWatch no POC (Recommended)
- B. Dashboard CloudWatch com métricas de infraestrutura

[Answer]: A

## Q4 — Alarmes e notificação
- A. **Nenhum alarme no POC**; lista de alarmes propostos documentada em `alarms.md` (Recommended)
- B. Alarmes + SNS agora

[Answer]: A

## Q5 — Rastreamento
- A. **X-Ray ativo nas Lambdas e no router, só `botocore`** (já implementado) (Recommended)
- B. Instrumentar também chamadas HTTP — grava o token do Telegram nos traços

[Answer]: A

## Q6 — Anomalias
- A. **Manter o `anomaly-detector` a cada minuto com limiar 0,7** e revisar com os alertas reais (Recommended)
- B. Recalibrar agora sem dados reais suficientes

[Answer]: A

---

## Consolidated Summary Confirmation

Resumo do estágio Observability Setup:
1. Observabilidade do POC = logs (14 d/7 d), X-Ray ativo e dashboard de negócio; sem alarmes, SNS nem métricas customizadas (NFR5.3 parcial, aceito).
2. SLOs da POC definidos e medidos manualmente; latência p90 < 10 s não atingida hoje (8–15 s).
3. X-Ray só no `botocore`, para não gravar o token do Telegram.
4. Detecção de anomalias a cada minuto, limiar 0,7; sem notificação ativa.
5. Alarmes e dashboard CloudWatch ficam como evolução pós-POC, documentados.

- Looks correct
- Request changes

[Answer]: Looks correct
