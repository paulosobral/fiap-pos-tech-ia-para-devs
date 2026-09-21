# Code Summary — u4-async-ingest

> Estágio Code Generation (Construction). Escopo `feature`, estratégia `standard`, metodologia `test-after`.

## Files created/modified

**Aplicação (`apps/contact-ingest/`, estrutura do PRD §7.4):**
- `handler.py` — handler Lambda SES (wiring de produção com boto3/env e DI por construtor); e-mail inválido → `drop` explícito (o evento nunca derruba por entrada inválida); falha transitória → `PendingRetryError` sinaliza o retry async da Lambda; `ROUTER_BASE_URL` **e** `INTERNAL_SECRET_TOKEN` obrigatórias via `_env` (fail-fast)
- `service/contact_ingest.py` — orquestrador ContactIngest: parse → dedupe → abrir sessão → re-injetar 1ª mensagem no ConversationRouter; outcomes `ok`/`retry`/`drop`; rollback da marca de dedupe em falha transitória — rollback impossível → marca movida para estado QUARANTINE (poison explícito, liberada no reprocesso); chave de dedupe nunca vazia (`messageId` ou hash de conteúdo sha256); logs estruturados JSON PII-safe com exceções sanitizadas (apenas tipo) (`log_event`/`exc_type`) (NFR5.1 + NFR2.1)
- `service/email_parser.py` — Protocol `EmailParser` + `HeuristicEmailParser` (FR12.2): extrai nome/e-mail/telefone de formatos de portal (corpo rotulado `Nome:/E-mail:/Telefone:`, padrão de assunto `Novo lead: ...`, cabeçalho From); decodifica corpo MIME cru (`ses.mail.content` base64) via stdlib `email`; sem e-mail extraível → `None` (drop); texto da mensagem truncado a 2000 chars; logs de erro sem conteúdo da exceção (PII-safe)
- `service/router_gateway.py` — Protocol `RouterGateway` + `HttpRouterGateway` (FR12.3): re-injeção alinhada ao contrato real da u1 `POST {ROUTER_BASE_URL}/internal/inbound-text` com header obrigatório `X-Internal-Secret` e body exato `{telegram_user_id: int, session_id: str, text: str}` (todos obrigatórios na u1; qualquer >= 300 → `RouterError`); sem secret configurado falha alta e cedo (`ValueError`), nunca envia sem autenticação; nenhum import de internals do `apps/conversation-router`
- `infra/session_store.py` — SessionWriter (Contrato 5): abre Lead (`PK=LEAD#<id>`, `SK=PROFILE`) + Conversation (`SK=CONV#<session_id>`) espelhando o padrão de escrita do SessionStore do u1; conversa nasce em `greeting`, `pii_masked/consent_recorded=False`, TTL 90 dias; contato extraído no `context`; log de erro sem conteúdo da exceção
- `infra/dedupe_store.py` — DedupeStore: put condicional `attribute_not_exists(message_id)` com `status=INGESTED` (anti spam/duplicata), grava `received_at` + domínio do remetente; `delete` best-effort para rollback; `mark_quarantine`/`take_quarantined` implementam o estado QUARANTINE (distinção "duplicata de ingest concluída" vs "marca obsoleta de tentativa falha"); logs de erro sem conteúdo da exceção

**Testes:** `tests/unit/test_contact_ingest.py` (21), `test_email_parser.py` (10), `test_session_store.py` (8), `test_dedupe_store.py` (12), `test_router_gateway.py` (7), `tests/integration/test_ingest_pipeline.py` (11)

**Config:** `apps/contact-ingest/requirements.txt`, `tests/conftest.py` (bootstrap de path). `pyproject.toml` da raiz, `apps/{conversation-router,voice-adapter,crm-adapter}/**` intocados; nenhuma outra config de raiz modificada.

## Key implementation decisions

- DI por construtor em todos os componentes (parser, dedupe, sessões, router) — a suíte roda sem AWS real, sem rede e sem dependências além do `.venv` existente; o evento SES é JSON puro.
- Semântica de entrega SES→Lambda (NFR4.1): SES não tem `batchItemFailures`/redrive como SQS. `drop` explícito para e-mail não parseável, duplicata e mail ausente (nunca derruba o evento, log em cada caso). Falha transitória (`DedupeError`/`SessionError`/`RouterError`/inesperada) vira outcome `retry` e o `handler()` levanta `PendingRetryError` → retry async da Lambda (2 tentativas); o reprocesso é idempotente via rollback da marca de dedupe.
- Dedupe-first (put condicional por `message_id`) garante uma única sessão por mensagem; rollback (`delete` best-effort) nas falhas posteriores evita perder o lead quando o retry async reprocessa — rollback impossível move a marca para `QUARANTINE` (nunca drop silencioso no reprocesso); `messageId` ausente → chave por hash de conteúdo (nunca key vazia).
- Limitação da POC aceita: falha do router após abrir a sessão deixa a sessão da 1ª tentativa órfã (o retry abre nova); ajuste fino (reuso de sessão no reprocesso) fica para Build and Test.
- `telegram_user_id` do Contrato 5 derivado por hash determinístico de (e-mail, lead_id) — leads de portal não têm conta Telegram; id único por inquiry, sem colisão com ids reais do Telegram, estável para auditoria.
- PII-safe em profundidade (NFR2.1): nome/e-mail/telefone/corpo nunca entram em logs — apenas `message_id`, `lead_id`, `session_id`, outcome; o dedupe grava apenas o **domínio** do remetente; `caplog` prova a ausência de PII nos logs estruturados.
- Contato extraído (FR12.2) gravado em `Conversation.context.contact` (campo `context` do Contrato 5); Lead fica minimalista conforme o schema compartilhado — a extração completa/PII real no LLM continua responsabilidade do SecurityLayer do u1.
- Parser 100% stdlib (`email`, `base64`, `re`): tolerante a formatos de portal (corpo rotulado, assunto, From), sem dependência externa.
- Logs estruturados JSON (`log_event`) no mesmo estilo do u2/u3 (NFR5.1).

## Test coverage summary

- 71 testes (58 unit + 11 integração) — todos verdes.
- Cobertura `apps/contact-ingest`: **100%** (TOTAL: 1009 stmts, 0 miss — piso 80% ✓).
- Comando unit-scoped registrado em `unit-test-instructions.md` (com `COVERAGE_FILE` isolado entre suítes); suítes re-verificadas verdes após a rodada de fix: u1 (116 testes), u2 (75), u3 (117). `compileall` OK.

## Deviations from the plan

- SES não tem `batchItemFailures`/DLQ como o SQS dos contratos 3/4: o outcome `retry` foi mapeado para `PendingRetryError` no `handler()` (retry async da Lambda), com idempotência via rollback do dedupe — o mecanismo primário de redrive do Contract 4 não se aplica a SES.
- `telegram_user_id` pseudo-derivado por hash (Contrato 5 exige inteiro) não estava especificado; necessário para leads de portal sem conta Telegram.
- Rollback da marca de dedupe (`delete` best-effort) em falha transitória não estava especificado; incluído para tornar o retry async idempotente sem perder leads.
- Sessão órfã quando o router falha após a abertura (retry abre nova sessão) — limitação documentada, aceita na POC.
- IaC (Terraform) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.

## Deviations da rodada de fix (iteração 2 do gate)

- **NFR4.1 marcado como Deferred no traceability.json** (R-02): a claim "componentes serverless com DLQ" NÃO é materializada por código/config nesta unidade — o destino on-failure (DLQ/Lambda destination) para o retry async esgotado da SES→Lambda é configuração de IaC da Lambda e entra no estágio Build and Test (código `handler.py` já sinaliza a falha que a DLQ estaciona).
- **NFR2.1 re-mapeado para o código real** (R-06): target agora é `apps/contact-ingest/service/contact_ingest.py` (materialização do logging PII-minimizado: `log_event` com apenas `message_id`/`session_id`/`lead_id`, `_source_domain` grava só o domínio, `exc_type` sanitiza exceções) — alinhado a NFR5.1; a máscara-antes-do-LLM é preservada transitivamente via SecurityLayer do u1 (esta unidade não envia texto a LLM).
- **Gateway alinhado ao contrato interno real da u1** (R-01): a u1 implementou `POST /internal/inbound-text` (autenticação `X-Internal-Secret`, body exato `{"telegram_user_id": int, "session_id": str, "text": str}` — ver seção "Superfície de contrato interno (real)" do code-summary da u1). O `HttpRouterGateway` da u4 foi ajustado ao contrato exato: secret obrigatório (fail-fast `ValueError` sem env — nada de envio sem autenticação, que viraria 401→retry→drop), body/header testados contra o shape exato; nenhum import de internals da u1.
- **Quarantine no dedupe quando rollback falha** (R-03): falha pós-dedupe + rollback OK → chave liberada (reprocesso ingere); rollback falho → `mark_quarantine` move a marca para estado `QUARANTINE` com motivo PII-safe (`rollback_failed:<TipoExceção>`) + logs de erro — e no reprocesso `take_quarantined` distingue "duplicata de ingest concluída" (drop) de "marca obsoleta de tentativa falha" (libera e reingere). Se até a quarentena falhar, log explícito `dedupe_quarantine_failed` (caminho de diagnóstico; nenhum drop silencioso). Coberto por testes unitários e end-to-end.
- **Chave de dedupe nunca vazia** (R-04): `messageId` ausente/vazio derivava a chave por sha256 de (remetente, assunto, corpo normalizado) — e-mails distintos sem `messageId` têm chaves distintas (não colapsam); conteúdo idêntico continua dedupado. `sha256:<hex>` no lugar da key vazia.
- **Exceções sanitizadas nos logs de erro** (R-05): todos os paths de erro (`contact_ingest`, `dedupe_store`, `session_store`, `email_parser`, `router_gateway`) registram apenas o nome da classe da exceção (`exc_type`) — nada de `str(exc)`/repr verbatim (PII de payloads em exceções inesperadas não vaza ao CloudWatch). Teste caplog de path de falha prova ausência de PII.
- **`INTERNAL_SECRET_TOKEN` obrigatória via `_env`** (R-07): mesma assimetria de config de `ROUTER_BASE_URL` eliminada — env ausente em produção = falha alta e cedo no wiring do handler (testada).
