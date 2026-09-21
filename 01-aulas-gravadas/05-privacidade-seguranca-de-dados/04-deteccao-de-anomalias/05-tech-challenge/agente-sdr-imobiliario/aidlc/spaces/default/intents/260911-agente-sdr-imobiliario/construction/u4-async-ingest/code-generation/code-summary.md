# Code Summary — u4-async-ingest

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/contact-ingest/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda SES (wiring de produção com boto3/env e DI por construtor); e-mail inválido → `drop` explícito (o evento nunca derruba por entrada inválida); falha transitória → `PendingRetryError` sinaliza o retry async da Lambda
- `service/contact_ingest.py` — orquestrador ContactIngest: parse → dedupe → abrir sessão → re-injetar 1ª mensagem no ConversationRouter; outcomes `ok`/`retry`/`drop`; rollback da marca de dedupe em falha transitória; logs estruturados JSON PII-safe (`log_event`) (NFR5.1 + NFR2.1)
- `service/email_parser.py` — Protocol `EmailParser` + `HeuristicEmailParser` (FR12.2): extrai nome/e-mail/telefone de formatos de portal (corpo rotulado `Nome:/E-mail:/Telefone:`, padrão de assunto `Novo lead: ...`, cabeçalho From); decodifica corpo MIME cru (`ses.mail.content` base64) via stdlib `email`; sem e-mail extraível → `None` (drop); texto da mensagem truncado a 2000 chars
- `service/router_gateway.py` — Protocol `RouterGateway` + `HttpRouterGateway` (FR12.3): re-injeção `POST {ROUTER_BASE_URL}/internal/inbound-text` com `X-Internal-Secret`, mesmo padrão do u2; nenhum import de internals do `apps/conversation-router`
- `infra/session_store.py` — SessionWriter (Contrato 5): abre Lead (`PK=LEAD#<id>`, `SK=PROFILE`) + Conversation (`SK=CONV#<session_id>`) espelhando o padrão de escrita do SessionStore do u1; conversa nasce em `greeting`, `pii_masked/consent_recorded=False`, TTL 90 dias; contato extraído no `context`
- `infra/dedupe_store.py` — DedupeStore: put condicional `attribute_not_exists(message_id)` (anti spam/duplicata), grava `received_at` + domínio do remetente; `delete` best-effort para rollback

**Testes:** `tests/unit/test_contact_ingest.py` (11), `test_email_parser.py` (10), `test_session_store.py` (8), `test_dedupe_store.py` (6), `test_router_gateway.py` (6), `tests/integration/test_ingest_pipeline.py` (9)

**Config:** `apps/contact-ingest/requirements.txt`, `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz, `apps/{conversation-router,voice-adapter,crm-adapter}/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- DI por construtor em todos os componentes (parser, dedupe, sessões, router) — a suíte roda sem AWS real, sem rede e sem dependências além do `.venv` existente; o evento SES é JSON puro.
- Semântica de entrega SES→Lambda (NFR4.1): SES não tem `batchItemFailures`/redrive como SQS. `drop` explícito para e-mail não parseável, duplicata e mail ausente (nunca derruba o evento, log em cada caso). Falha transitória (`DedupeError`/`SessionError`/`RouterError`/inesperada) vira outcome `retry` e o `handler()` levanta `PendingRetryError` → retry async da Lambda (2 tentativas); o reprocesso é idempotente via rollback da marca de dedupe.
- Dedupe-first (put condicional por `message_id`) garante uma única sessão por mensagem; rollback (`delete` best-effort) nas falhas posteriores evita perder o lead quando o retry async reprocessa.
- Limitação da POC aceita: falha do router após abrir a sessão deixa a sessão da 1ª tentativa órfã (o retry abre nova); ajuste fino (reuso de sessão no reprocesso) fica para Build and Test.
- `telegram_user_id` do Contrato 5 derivado por hash determinístico de (e-mail, lead_id) — leads de portal não têm conta Telegram; id único por inquiry, sem colisão com ids reais do Telegram, estável para auditoria.
- PII-safe em profundidade (NFR2.1): nome/e-mail/telefone/corpo nunca entram em logs — apenas `message_id`, `lead_id`, `session_id`, outcome; o dedupe grava apenas o **domínio** do remetente; `caplog` prova a ausência de PII nos logs estruturados.
- Contato extraído (FR12.2) gravado em `Conversation.context.contact` (campo `context` do Contrato 5); Lead fica minimalista conforme o schema compartilhado — a extração completa/PII real no LLM continua responsabilidade do SecurityLayer do u1.
- Parser 100% stdlib (`email`, `base64`, `re`): tolerante a formatos de portal (corpo rotulado, assunto, From), sem dependência externa.
- Logs estruturados JSON (`log_event`) no mesmo estilo do u2/u3 (NFR5.1).

## Test coverage summary

- 50 testes (41 unit + 9 integração) — todos verdes.
- Cobertura `apps/contact-ingest`: **99.74%** (TOTAL: 773 stmts, 2 miss — piso 80% ✓). Misses: `handler.py:20` (raise defensivo de `_env`) e `contact_ingest.py:132` (log de rollback best-effort).
- Comando unit-scoped registrado em `unit-test-instructions.md`; suítes re-verificadas verdes após o trabalho: u1 (50 testes), u2 (58), u3 (86). `compileall` OK.

## Deviations from the plan

- SES não tem `batchItemFailures`/DLQ como o SQS dos contratos 3/4: o outcome `retry` foi mapeado para `PendingRetryError` no `handler()` (retry async da Lambda), com idempotência via rollback do dedupe — o mecanismo primário de redrive do Contract 4 não se aplica a SES.
- `telegram_user_id` pseudo-derivado por hash (Contrato 5 exige inteiro) não estava especificado; necessário para leads de portal sem conta Telegram.
- Rollback da marca de dedupe (`delete` best-effort) em falha transitória não estava especificado; incluído para tornar o retry async idempotente sem perder leads.
- Sessão órfã quando o router falha após a abertura (retry abre nova sessão) — limitação documentada, aceita na POC.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
