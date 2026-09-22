# Memory — deployment-execution

**Estágio:** deployment-execution · **Data:** 2026-09-21

## O que foi feito

- `start.sh` executado ponta a ponta até `SMOKE OK` (EXIT=0) — deploy real na AWS conta 144842881551/us-east-1.
- 7 correções iterativas durante o deploy (ver deployment-log.md §2): repackage módulo Lambda, secret vazio, policy KMS, voice zip 128MB→pequeno (POC sem whisper), attach IAM do módulo (`number_of_policies`), imagem ECS vazia→placeholder público, smoke Bearer.
- API `https://izlmelzop8.execute-api.us-east-1.amazonaws.com` — /health 200, webhook 401, /api/kpis 200.

## Decisões registradas

- **B. Streamlit Fargate mini** — 0.25/0.5GB, sem ALB, IP público, escala 0 fora de 09:00–17:00 BRT (~US$0.95/mês).
- Voice DEGRADADO no POC (voz → DLQ); pós-POC: layer/Transcribe.
- Bot Telegram inativo até secret preenchida (token não vai no código/state).
- /api/kpis valida presença de Bearer (POC); valor pós-POC.

## Estado técnico

- Imagem ECR: `144842881551.dkr.ecr.us-east-1.amazonaws.com/sdr-dashboard-ui:poc-20260921185026`.
- Terraform state local em infra/; teardown `./stop.sh`.
- 7 Lambdas empacotadas (dist/), mappings/EventBridge/queues/tables/KMS/secrets provisionados.
