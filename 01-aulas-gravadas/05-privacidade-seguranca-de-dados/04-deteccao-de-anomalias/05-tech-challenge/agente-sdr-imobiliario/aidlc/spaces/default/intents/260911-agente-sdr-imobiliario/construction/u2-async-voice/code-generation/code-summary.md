# Code Summary — u2-async-voice

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/voice-adapter/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda SQS (resposta parcial `batchItemFailures`, nunca derruba o lote); wiring de produção com boto3/env e DI por construtor
- `service/voice_adapter.py` — orquestrador VoiceAdapter: valida schema do Contract 3 → sessão (Contract 5) → download → conversão → transcrição → PII → re-injeção; outcomes `ok`/`retry`/`drop`; fallback Telegram; logs estruturados JSON (NFR5.1)
- `service/transcriber.py` — interface `Transcriber` (Protocol) + `WhisperTranscriber` (ffmpeg→WAV 16k mono, faster-whisper PT-BR, import protegido, `model_factory` injetável, modelo carregado lazy)
- `service/telegram_gateway.py` — Telegram Bot API (getFile/download/sendMessage) com http client injetado
- `service/router_gateway.py` — interface `RouterGateway` (Protocol) + `HttpRouterGateway` (re-injeção via POST `/internal/inbound-text`)
- `service/pii.py` — PiiMasker (NFR2.1/NFR2.2: NOME/EMAIL/TELEFONE/CNPJ → placeholders)
- `infra/session_store.py` — SessionLookup (leitor Contract 5, GSI `session-index`, valida `telegram_user_id`)

**Testes:** `tests/unit/test_voice_adapter.py` (12), `test_telegram_gateway.py` (10), `test_transcriber.py` (10), `test_router_gateway.py` (5), `test_session_store.py` (5), `test_pii.py` (6), `tests/integration/test_voice_adapter_pipeline.py` (10)

**Config:** `apps/voice-adapter/requirements.txt`, `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz e `apps/conversation-router/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- DI por construtor em todos os componentes (telegram, transcriber, router, sessions, masker) — a suíte roda sem AWS real, sem faster-whisper e sem ffmpeg.
- Semântica SQS partial-batch: `retry` → `batchItemFailures` (redrive SQS → DLQ, Contract 3); `drop` explícito (schema inválido, sessão ausente, áudio inacessível, conversão falha, transcript vazio) sai do lote **sem** falha; exceção inesperada → `retry` (o lote nunca quebra — NFR4.1).
- PII: transcript mascarado antes da re-injeção e antes de qualquer log (NFR2.1/NFR2.2 em profundidade; mascaramento é idempotente em relação ao SecurityLayer do u1, que re-mascara no fluxo).
- Import de faster-whisper protegido no nível de módulo; modelo carregado lazy no primeiro uso; `model_factory` injetável para testes/POC.
- Re-injeção desacoplada: `RouterGateway` injetado — nenhum import de internals de `apps/conversation-router`.
- Fallback de voz: mensagem amigável via Telegram para áudio inválido/inacessível, transcript vazio e sessão ausente (constraint da unidade: nunca derrubar o lote, sempre responder).
- Logs estruturados JSON (`log_event`) sem texto bruto do transcript — apenas contagens/ids (NFR5.1 + PII-safe).

## Test coverage summary

- 58 testes (48 unit + 10 integração) — todos verdes.
- Cobertura `apps/voice-adapter`: **99.86%** (piso 80% ✓). Única linha não coberta: `service/transcriber.py:15` (import protegido de `faster_whisper` — exige o pacote pesado instalado).
- Comando unit-scoped registrado em `unit-test-instructions.md`.

## Deviations from the plan

- faster-whisper/ffmpeg não são exigidos pela suíte (imports protegidos + `subprocess.run` mockado); a implementação real está presente e exercível localmente, mas a POC roda com stubs (`model_factory`/MagicMock). Consequência: `transcriber.py:15` fica fora da cobertura (99.86% ≥ piso de 80%).
- Re-injeção implementada como `HttpRouterGateway` → `POST {ROUTER_BASE_URL}/internal/inbound-text` com header `X-Internal-Secret`. Não existia contrato HTTP explícito para re-injeção de texto (o Contract 3 cobre apenas o SQS de voz); o endpoint interno do ConversationRouter é presumido e o gateway injetado permite trocar o mecanismo sem tocar no orquestrador. Ajuste final do endpoint/contrato fica para Build and Test.
- SessionLookup consulta a GSI `session-index` (atributo `session_id`) da tabela de sessões — GSI a provisionar na infraestrutura (Terraform fora do escopo deste estágio).
- u2 não escreve no DynamoDB (Contract 5 é somente leitura para esta unidade); o transcript persiste no u1 via re-injeção como mensagem inbound de texto.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
