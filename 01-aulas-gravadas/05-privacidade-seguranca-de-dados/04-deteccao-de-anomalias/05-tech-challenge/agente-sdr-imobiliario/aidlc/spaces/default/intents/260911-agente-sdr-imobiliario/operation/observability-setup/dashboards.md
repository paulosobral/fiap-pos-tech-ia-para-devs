# Dashboards — Agente SDR Imobiliário (POC)

## O que existe hoje
- **Dashboard de negócio (Streamlit, ECS `dashboard-ui`)**: funil de leads, leads com imóvel escolhido e valor, anomalias (alertas), KPIs de `GET /api/kpis`. É a visão operacional do gestor, com login Cognito. Acesso pelo IP público da subida (ver `logs/start-latest.log`).
- **CloudWatch Logs**: um log group por Lambda e por serviço ECS (retenção 14 dias nas Lambdas, 7 dias no ECS).
- **X-Ray**: mapa de serviços e traços das Lambdas e do router (ver `tracing-config.md`).

## O que não existe (decisão do POC)
- Não há dashboard do CloudWatch nem métricas customizadas: o código não emite `PutMetricData` nem EMF. As métricas `ResponseTimeP90` e `CostMonthly`, citadas no design, nunca foram emitidas (NFR5.3 parcial).
- Motivo: custo e tempo do hackathon; o ambiente é efêmero (destruído pelo `stop.sh`), então um dashboard persistente teria pouco uso.

## Evolução pós-POC
1. Dashboard CloudWatch com: latência de turno do router (p50/p90), erros por Lambda, profundidade das filas e da DLQ, tasks ECS em execução.
2. Emitir `ResponseTimeP90` e custo de LLM por conversa (a LLM já devolve o uso de tokens).
