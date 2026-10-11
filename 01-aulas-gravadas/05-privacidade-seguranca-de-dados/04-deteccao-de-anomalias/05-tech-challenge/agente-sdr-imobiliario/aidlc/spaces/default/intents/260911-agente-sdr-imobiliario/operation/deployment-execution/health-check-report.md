# Health Check Report — agente SDR imobiliário (POC)

**Data:** 2026-09-21 · **API:** `https://izlmelzop8.execute-api.us-east-1.amazonaws.com`

## Status por componente

| Componente | Status | Detalhe |
|------------|--------|---------|
| HTTP API + /health | ✅ OK | 200 `{"status":"ok"}` |
| Webhook Telegram (u1) | ✅ OK | valida secret; 401 sem header |
| conversation-router | ✅ OK | /health responde; rotas /internal protegidas por INTERNAL_SECRET_TOKEN |
| contact-ingest (u4) | ✅ OK | mapping criado; POC alimenta via `aws sqs send-message sdr-ingest-queue` |
| crm-adapter (u3) | ✅ OK | mapping criado; CSV path padrão /tmp/crm.csv |
| dashboard-api (u7) | ✅ OK | /api/kpis 200 (bearer presente) |
| anomaly-detector (u5) | ✅ agendado | EventBridge rate(1 minute) |
| followup (u6) | ✅ agendado | EventBridge rate(1 day) |
| **voice-adapter (u2)** | ⚠️ **DEGRADADO** | POC empacota sem faster-whisper/ffmpeg/ctranslate2 (limite 50MB do zip direto); mensagens de voz falham → DLQ `sdr-voice-dlq`. Evolução pós-POC: layer + Amazon Transcribe |
| **dashboard-ui (ECS)** | 🟦 escala 0 fora da janela | desired_count 0 (09:00–17:00 BRT escala 0→1→0); IP público efêmero por task; validar via Console ECS |
| **bot Telegram** | 🔒 inativo até ativação | secret `sdr/tg-bot-token` vazia; para ativar: definir token no Secrets Manager (ou `-var=telegram_bot_token=...`) e re-apply |

## Endpoints verificados

- `GET /health` → 200 (código em apps/conversation-router/handler.py, antes das rotas /internal).
- `GET /api/kpis` → 200 com Bearer (POC); 401 sem header.
- `POST /webhook/telegram` → 401 sem `X-Telegram-Bot-Api-Secret-Token`.

## Itens para evolução pós-POC

1. Validar valor do Bearer em /api/kpis contra o secret interno.
2. Transcrição de voz (layer faster-whisper ou Amazon Transcribe) + remover estado DEGRADADO.
3. Backend terraform remoto (S3+DynamoDB lock) em vez de state local.
4. Dashboard: dominio/HTTPS (ALB só se necessário — custo ~US$16/mês) e COGNITO_LOGIN_URL.

## Atualização (2026-10-11)
Estado após a subida de 10/10 (`logs/start-20261010-130721.log`):
- **voice-adapter**: não está mais degradado; roda como container ECS com faster-whisper e consome `sdr-voice-queue`.
- **bot Telegram**: webhook configurado pelo `start.sh` (`Telegram: OK`); o token vem de `secrets.local.env` e do Secrets Manager.
- **conversation-router e dashboard-ui**: serviços ECS com 3/3 tasks em execução ao final do deploy; IP público efêmero, janela 09:00–18:00 BRT.
- **dashboard-api**: `/api/kpis` exige token Cognito (smoke OK).
- Ainda sem confirmação numa subida real: X-Ray, SSM `/sdr/bot-name`, índice `lead-index` de alertas e o fluxo de encerramento (alterações posteriores à última subida).
