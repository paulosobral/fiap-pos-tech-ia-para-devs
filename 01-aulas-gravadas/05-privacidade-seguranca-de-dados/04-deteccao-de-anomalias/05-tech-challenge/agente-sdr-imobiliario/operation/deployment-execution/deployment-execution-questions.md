# Deployment Execution — Questions (Consolidated)

**Estágio:** deployment-execution · **Intent:** 260911-agente-sdr-imobiliario · **Data:** 2026-09-21

Perguntas geradas do trabalho de deployment-execution para consolidação com os decisions.

## Q1 — Smoke do pipeline fechou 3/3 (health 200, webhook 401 sem secret, kpis 200 com Bearer)?

**Evidência:** smoke-test-results.md; start.sh fase [6/6] exit 0.

## Q2 — Voice-adapter permanece DEGRADADO no POC (pacote sem faster-whisper; voz vai para DLQ) e a evolução (layer/Transcribe) fica para pós-POC?

**Evidência:** health-check-report.md; limite de 50MB do zip direto da Lambda (~120MB do stack de transcrição).

## Q3 — Dashboard-ui opera escala 0 fora da janela 09:00–17:00 BRT (custo ~US$0.95/mês) e validação manual na janela via Console ECS?

**Evidência:** deployment-log.md §3; decisa do humano (opção B — Fargate mini sem ALB).

## Q4 — Bot Telegram fica inativo até o humano preencher a secret `sdr/tg-bot-token` e re-aplicar (não embutimos token no código/state)?

**Evidência:** health-check-report.md; secret_version condicional (`count = token != ""`).

## Consolidated Summary Confirmation

Consolidated work summary built from the question flow and decisions of this stage. Does this look correct? Choose "Looks correct" or "Request changes".

- Q1: Looks correct
- Q2: Looks correct
- Q3: Looks correct
- Q4: Looks correct

[Answer]:
