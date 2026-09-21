# Code Summary — u2-async-voice

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/voice-adapter/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda SQS (resposta parcial `batchItemFailures`, nunca derruba o lote); wiring de produção com boto3/env e DI por construtor; `get_transcriber()` singleton lazy em nível de módulo (o modelo Whisper sobrevive entre invocações warm); envs `TELEGRAM_BOT_TOKEN`, `ROUTER_BASE_URL` e `INTERNAL_SECRET_TOKEN` obrigatórias via `_env`
- `service/voice_adapter.py` — orquestrador VoiceAdapter: valida schema do Contract 3 → sessão (Contract 5) → download → conversão → transcrição → PII → re-injeção; outcomes `ok`/`retry`/`drop`; fallback Telegram; logs estruturados JSON (NFR5.1)
- `service/transcriber.py` — interface `Transcriber` (Protocol) + `WhisperTranscriber` (ffmpeg→WAV 16k mono com `timeout` configurável, faster-whisper PT-BR, import protegido, `model_factory` injetável, modelo carregado lazy; erro de áudio → `AudioConversionError` (drop), erro de ambiente ffmpeg (ausente/travado) → `TranscriptionError` com causa raiz (retry/DLQ))
- `service/telegram_gateway.py` — Telegram Bot API (getFile/download/sendMessage) com http client injetado
- `service/router_gateway.py` — interface `RouterGateway` (Protocol) + `HttpRouterGateway` (re-injeção no contrato interno real da U1: POST `/internal/inbound-text`, header `X-Internal-Secret`, body `{telegram_user_id, session_id, text}`; secret obrigatório — fail-fast no wiring)
- `service/pii.py` — PiiMasker espelhando a SecurityLayer oficial da U1 (mesmos regexes NOME/EMAIL/CNPJ + lista controlada `COMMON_FIRST_NAMES` para nome único; TELEFONE estendido a formatos locais sem +55)
- `infra/session_store.py` — SessionLookup (leitor Contract 5): resolve o lead pela GSI `telegram-user-index` e lê a conversa pela chave composta `LEAD#<lead_id>/CONV#<session_id>` — acesso real da U1, sem GSI inventada

**Testes:** `tests/unit/test_voice_adapter.py` (13), `test_telegram_gateway.py` (10), `test_transcriber.py` (13), `test_router_gateway.py` (6), `test_session_store.py` (7), `test_pii.py` (13), `tests/integration/test_voice_adapter_pipeline.py` (13)

**Config:** `apps/voice-adapter/requirements.txt`, `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz e `apps/conversation-router/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- DI por construtor em todos os componentes (telegram, transcriber, router, sessions, masker) — a suíte roda sem AWS real, sem faster-whisper e sem ffmpeg.
- Semântica SQS partial-batch: `retry` → `batchItemFailures` (redrive SQS → DLQ, Contract 3); `drop` explícito (schema inválido, sessão ausente, áudio inacessível, conversão falha, transcript vazio) sai do lote **sem** falha; exceção inesperada → `retry` (o lote nunca quebra — NFR4.1).
- PII: transcript mascarado antes da re-injeção e antes de qualquer log (NFR2.1/NFR2.2 em profundidade); o PiiMasker espelha a SecurityLayer oficial da U1 (mesmos regexes + lista controlada de nomes) para os dois lados evoluírem juntos; mascaramento é idempotente em relação ao re-mascaramento da U1 no fluxo.
- Import de faster-whisper protegido no nível de módulo; modelo carregado lazy no primeiro uso; `model_factory` injetável para testes/POC; o transcriber é singleton de nível de módulo no `handler.py` (`get_transcriber()`) para sobreviver entre invocações warm (R-03).
- Classificação de falhas da transcrição: erro de ÁUDIO (`AudioConversionError`) → drop + fallback; erro de AMBIENTE/TRANSIENTE (ffmpeg ausente `OSError`, ffmpeg travado `TimeoutExpired`) → `TranscriptionError` com causa raiz → retry/redrive → DLQ (NFR4.1, R-06); `subprocess.run` do ffmpeg roda com `timeout` configurável (default 20s, dentro da visibility timeout de 30s do Contract 3).
- Re-injeção desacoplada: `RouterGateway` injetado — nenhum import de internals de `apps/conversation-router`; o shape HTTP segue exatamente o contrato interno real da U1 (rota, header e body documentados em "Superfície de contrato interno (real)" do code-summary da U1).
- Fallback de voz: mensagem amigável via Telegram para áudio inválido/inacessível, transcript vazio e sessão ausente (constraint da unidade: nunca derrubar o lote, sempre responder).
- Logs estruturados JSON (`log_event`) sem texto bruto do transcript — apenas contagens/ids (NFR5.1 + PII-safe).

## Test coverage summary

- 75 testes (62 unit + 13 integração) — todos verdes.
- Cobertura `apps/voice-adapter`: **99.88%** (piso 80% ✓). Única linha não coberta: `service/transcriber.py:15` (import protegido de `faster_whisper` — exige o pacote pesado instalado).
- Comando unit-scoped registrado em `unit-test-instructions.md`.

## Deviations from the plan

- faster-whisper/ffmpeg não são exigidos pela suíte (imports protegidos + `subprocess.run` mockado); a implementação real está presente e exercível localmente, mas a POC roda com stubs (`model_factory`/MagicMock). Consequência: `transcriber.py:15` fica fora da cobertura (99.88% ≥ piso de 80%).
- Re-injeção implementada como `HttpRouterGateway` → `POST {ROUTER_BASE_URL}/internal/inbound-text` com header `X-Internal-Secret` e body `{telegram_user_id, session_id, text}` — agora alinhada ao contrato interno REAL da U1 (implementado na rodada de fix da U1 e documentado na seção "Superfície de contrato interno (real)" do code-summary da U1); o gateway injetado continua substituível sem tocar no orquestrador.
- SessionLookup espelha o acesso REAL da tabela da U1: resolve o lead pela GSI `telegram-user-index` e lê a conversa pela chave composta `LEAD#<lead_id>/CONV#<session_id>` (mesmo padrão de `get_by_telegram_user`/`get_conversation` da U1). A GSI `session-index` NÃO existe e NÃO é mais referenciada — nenhuma infraestrutura nova é exigida desta unidade.
- u2 não escreve no DynamoDB (Contract 5 é somente leitura para esta unidade); o transcript persiste no u1 via re-injeção como mensagem inbound de texto.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.

## Deviations da rodada de fix (iteração 2 do gate)

- **R-01 (Critical) — gateway alinhado ao contrato interno real da U1:** o endpoint `POST /internal/inbound-text` agora EXISTE na U1 (u1 corrigida nesta rodada); o `HttpRouterGateway` foi conferido contra o handler da U1 — rota, método, body `{telegram_user_id: int, session_id: str, text: str}` e header `X-Internal-Secret` já casavam, mas o secret era OPCIONAL (`os.environ.get`): sem env, o request saía sem header → 401 → retry infinito → DLQ. FIX: `INTERNAL_SECRET_TOKEN` é OBRIGATÓRIA no wiring via `_env` (falha alta e cedo no cold start) e o gateway valida o secret no construtor (fail-fast, nenhum request sem header). Testes do shape exato do body e do header (incl. shape completo dos headers).
- **R-02 (Major) — lookup de sessão espelha o acesso real da U1:** a GSI `session-index` não existe (nem em contrato nem em infra). O `SessionLookup` agora resolve o lead pela GSI `telegram-user-index` (mesma query da U1: `telegram_user_id = :uid`, valor `{"N": str(...)}`) e lê a conversa com `get_item` na chave composta `LEAD#<lead_id>/CONV#<session_id>` (mesma forma de `get_conversation` da U1) — equivalente funcional de `get_by_telegram_user` + `get_conversation`. Sessão inexistente → `drop` + fallback Telegram (comportamento preservado). Zero GSI inventada; teste trava o índice e a chave exatos.
- **R-03 (Major) — Whisper singleton:** `WhisperTranscriber()` saiu do handler e virou singleton lazy de nível de módulo (`get_transcriber()`), mantendo o `model_factory` injetável para testes/POC; o modelo carregado sobrevive entre invocações warm (constraint < 15s por áudio). Teste prova UMA instanciação em duas invocações do handler + identidade do singleton.
- **R-04 (Minor) — PiiMasker espelha a SecurityLayer oficial:** mesmos regexes de NOME/EMAIL/CNPJ, mesma lista controlada `COMMON_FIRST_NAMES` e mesma passada de nome único (nome único capitalizado conhecido — "Pedro" — é mascarado; palavras comuns — "Podemos" — não; nomes de lugar 2-capitais — "São Paulo" — seguem o comportamento conservador da camada oficial, [NOME]); com prefixos ("João da Silva") o primeiro nome entra na lista controlada → "[NOME] da Silva" (comportamento da U1, sobrenome isolado é baixo risco de identificação). TELEFONE é um superconjunto deliberado do padrão oficial: além de +55, cobre formatos locais "11 91234-5678", "(11) 91234-5678" e "91234-5678" (defeito apontado no finding; a U1 só recebe texto já mascarado, então o superconjunto não gera divergência a jusante). Sem import cross-app (fronteira entre Lambdas mantida); cópia documentada para evoluir junto.
- **R-05 (Minor) — traceability honesto:** NFR2.5 (retenção TTL 90 dias) saiu de OK para **N/A** — retenção é responsabilidade de ESCRITA da U1 (`Conversation.ttl` em `entities.py`); o `SessionLookup` é somente leitura. FR1.2/FR1.3/Contract 3/Contract 5 re-verificados contra o código real pós-fix: agora OK de fato (endpoint real, acesso real), com justificativa da reverificação em cada row.
- **R-06 (Minor) — falhas de ambiente explícitas:** `OSError` (ffmpeg ausente) saiu de `AudioConversionError` (drop + ack silencioso) para `TranscriptionError` com causa raiz → `retry` → redrive → DLQ (NFR4.1); `subprocess.TimeoutExpired` (ffmpeg travado) idem; `subprocess.run` ganhou `timeout` configurável (default 20s) tratando o estouro como erro classificado. Erro de ÁUDIO real (`CalledProcessError`) permanece `AudioConversionError` → drop + fallback (correto). Testes de classificação (TranscriptionError que NÃO é AudioConversionError) + timeout aplicado/configurável.
