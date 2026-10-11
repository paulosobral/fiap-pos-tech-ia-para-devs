# Runbooks — Agente SDR Imobiliário (POC)

Procedimentos manuais (sem SSM Automation no POC). Onde ver cada sintoma: `observability-setup/log-queries.md`, X-Ray e o dashboard de negócio.

## RB1 — O bot não responde no Telegram
1. Conferir se é dentro da janela do ECS (09:00–18:00 BRT); fora dela o router fica com 0 tasks.
2. Ver se há task do `conversation-router` em execução (ECS) e o log group dele.
3. O IP do router muda a cada subida: o `start.sh` reaponta o API Gateway e o webhook. Se o IP mudou depois, rodar de novo o `start.sh` (ou subir o serviço de novo).
4. Conferir o webhook do Telegram (`getWebhookInfo`) e o token em `sdr/tg-bot-token`.

## RB2 — Respostas lentas ou com falha (LLM)
1. Procurar `429`/`fallback` nos logs. O DeepSeek devolve 429 com frequência; o tiering cai para Haiku 4.5 e depois Sonnet 4.5 (via OpenRouter).
2. Se todos os modelos falham, verificar a chave em `sdr/llm-api-key` e o saldo do OpenRouter.
3. Os modelos ficam em SSM (`/sdr/...`) e podem ser trocados sem novo deploy do código.

## RB3 — Lead não chega ao HubSpot
1. Ver a fila `crm` e a DLQ correspondente; a Lambda `crm-adapter` registra o erro.
2. O refresh token do HubSpot MCP é de uso único: se expirou ou foi gasto, rodar `scripts/hubspot_authorize.py` e atualizar `HUBSPOT_MCP_REFRESH_TOKEN` em `secrets.local.env`. O `stop.sh` salva o token vigente ao destruir o ambiente.
3. Reprocessar as mensagens da DLQ depois de corrigir.

## RB4 — Mensagem de voz falha
1. Ver `sdr-voice-dlq` e o log do `voice-adapter` (ECS, faster-whisper + ffmpeg).
2. Task fora da janela ou sem memória: reiniciar o serviço.

## RB5 — Dashboard não abre ou mostra gráficos vazios
1. O IP do dashboard muda a cada subida (ver `logs/start-latest.log`); fora da janela, as tasks estão em 0.
2. Login é no Cognito (`USER_PASSWORD_AUTH`); conferir o usuário.
3. Sem dados, o gráfico mostra a legenda de "sem dados" (não é erro).

## RB6 — Alertas de anomalia sem efeito ou erro de `ValidationException`
Conferir que o índice `lead-index` existe na tabela `sdr-alerts` (`tests/infra` protege isso). Sem o índice, a restrição de agendamento deixa de funcionar.

## RB7 — Deploy ou rollback
Seguir `operation/deployment-pipeline/rollback-runbook.md` (R1 a R4).
