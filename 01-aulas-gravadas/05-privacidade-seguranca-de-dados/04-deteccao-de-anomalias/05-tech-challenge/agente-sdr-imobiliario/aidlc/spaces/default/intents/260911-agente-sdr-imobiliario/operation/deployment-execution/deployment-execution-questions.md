# Deployment Execution — Questions (Consolidated)

**Estágio:** deployment-execution · **Intent:** 260911-agente-sdr-imobiliario · **Data:** 2026-09-21

Perguntas geradas do trabalho de deployment-execution para consolidação com os decisions.

## Q1 — Smoke do pipeline fechou 3/3 (health 200, webhook 401 sem secret, kpis 200 com Bearer)?

**Evidência:** smoke-test-results.md; start.sh fase [6/6] exit 0.

## Q2 — Voice-adapter roda em ECS Fargate (faster-whisper real, ~520MB) consumindo a fila SQS via worker long-polling, com escala 09:00–18:00 BRT (mesma janela do dashboard)?

**Resolução:** SIM. `faster-whisper` + `ctranslate2` + `ffmpeg` (~520MB) não cabem no limite de Lambda (250MB máximo somando layers/zip). O `voice-adapter` agora é um **worker ECS Fargate** (`service/sqs_worker.py`) que faz long-polling da `sdr-voice-queue`, transcreve com modelo `small` e reinjeta texto no router via `POST /internal/inbound-text`. A Lambda `sdr-voice-adapter` permanece declarada sem event source mapping (deprecated; fallback manual). Imagem ECR `sdr-voice-adapter` buildada/pushada no `start.sh` fase [5c/6].

**Evidência:** infra/ecs.tf (ECR + task def + service + autoscaling programático), infra/variables.tf (`voice_schedule_start/end` = `cron(0 12/21 * * ? *)` = 09:00–18:00 BRT), apps/voice-adapter/service/sqs_worker.py, apps/voice-adapter/Dockerfile.

## Q3 — Dashboard-ui opera escala 0 fora da janela 09:00–18:00 BRT (custo ~US$1.05/mês) e validação manual na janela via Console ECS?

**Evidência:** deployment-log.md §3; decisão do usuário: janela comercial 09:00–18:00 BRT (12:00–21:00 UTC).

## Q4 — Bot Telegram fica inativo até o humano preencher a secret `sdr/tg-bot-token` e re-aplicar (não embutimos token no código/state)?

**Evidência:** health-check-report.md; secret_version condicional (`count = token != ""`).

## Consolidated Summary Confirmation

Consolidated work summary built from the question flow and decisions of this stage. Does this look correct? Choose "Looks correct" or "Request changes".

- Q1: Looks correct
- Q2: Looks correct
- Q3: Looks correct
- Q4: Looks correct

[Answer]: Looks correct
