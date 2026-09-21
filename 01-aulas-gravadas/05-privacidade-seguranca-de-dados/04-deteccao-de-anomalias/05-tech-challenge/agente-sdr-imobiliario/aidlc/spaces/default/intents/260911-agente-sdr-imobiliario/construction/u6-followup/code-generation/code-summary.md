# Code Summary — u6-followup

## Files created/modified

Todos os arquivos são novos, sob `apps/followup/` (nenhum app existente foi tocado):

| Arquivo | Papel |
|---------|-------|
| `apps/followup/handler.py` | Entry point Lambda — tick diário do EventBridge; wiring DI completo (ConversationStore, FollowupStateStore, DuplicateGuard, CadenceCalculator, SilenceWindow, FollowupMessageBuilder, TelegramGateway); env: `SESSIONS_TABLE`, `FOLLOWUP_TABLE`, `TELEGRAM_BOT_TOKEN`, `FOLLOWUP_CADENCE_DAYS`, `SILENCE_WINDOW_START/END`, `TIMEZONE` |
| `apps/followup/requirements.txt` | Dependências de produção (`boto3>=1.34`, `requests>=2.31`) |
| `apps/followup/infra/conversation_store.py` | Leitor do Contrato 5: scan paginado `begins_with(SK, CONV#)` (conversas) e `begins_with(SK, PROFILE)` (mapa `lead_id → telegram_user_id/intent`); unmarshal de tipos/JSON; itens sem identificadores ignorados com log; erros → `ConversationStoreError` |
| `apps/followup/infra/followup_state.py` | Estado próprio de cadência (tabela `sdr-followup-state`, item `PK=LEAD#<id>, SK=FOLLOWUP`): `last_step`, `last_followup_at`, `next_step`, `done`; erros → `FollowupStateError` |
| `apps/followup/infra/silence_window.py` | FR8.3 — janela de silêncio configurável em horas do fuso local (padrão `America/Sao_Paulo`, override `TIMEZONE`, fim exclusivo), suporta janela que cruza meia-noite (`start > end`) e janela 24h (`start == end`); clock injetável; `from_env` |
| `apps/followup/service/cadence.py` | FR8.1 — `CadenceCalculator` (dias 2/5/9 configuráveis, `due_step` = maior dia já alcançado, `is_expired`, `next_step`); `parse_cadence_days`; `parse_iso8601`; clock injetável |
| `apps/followup/service/duplicate_guard.py` | Guarda anti-spam: `already_followed_up` (passo ≤ último enviado — idempotência do tick) e `lead_replied` (mensagem do lead posterior ao último follow-up); falha de leitura de estado de um lead é isolada por lead no orquestrador (tick continua) |
| `apps/followup/service/followup.py` | Orquestrador: conversa mais recente por lead → drop explícito (invalid_created_at / cadence_exhausted) → passo vencido → janela de silêncio (defer) → guarda → contexto PII-safe → mensagem PT-BR → envio → persistir próximo passo; processamento isolado por lead (falha de um lead não aborta o tick); guarda de resposta lê o campo `at` do schema real da U1; resumo com contadores e logs estruturados `log_event` |
| `apps/followup/service/message_builder.py` | FR8.2 — mensagens PT-BR por passo (2/5/9) personalizadas pela intenção (compra/locação/investimento); PII-safe por construção (recebe só `step`/`intent`/`state`) |
| `apps/followup/service/telegram_gateway.py` | Telegram Bot API `sendMessage` (mesmo mecanismo de envio da U1; o Contrato 1 cobre apenas o webhook de entrada) via http injetado (padrão voice-adapter); `GatewayError` sem URL/token do bot na mensagem; falha → estado não gravado, retry no próximo tick |
| `apps/followup/infra/structured_log.py` | `log_event` JSON compartilhado (NFR5.1) — formato único de log da unidade (service e infra), PII-safe por construção |
| `apps/followup/tests/conftest.py` | Bootstrap de path (`sys.path` → `apps/followup`), estilo flat u1–u5 |
| `apps/followup/tests/unit/test_*.py` | 7 suítes unitárias (cadence, silence_window, message_builder, telegram_gateway, followup_state, conversation_store, duplicate_guard, followup) |
| `apps/followup/tests/integration/test_followup_pipeline.py` | Pipeline ponta a ponta: handler + FakeDynamo + FakeHttp + env, idempotência, janela fechada, cadência custom, supressão por resposta, PII-safe nos logs, token ausente, payload não-dict |

## Key implementation decisions

- **Contrato 5 com dois scans**: a cadência precisa de `created_at` da conversa e do `telegram_user_id`/`intent` do Lead. U6 é reader do Contrato 5 — a leitura é feita com dois scans paginados (`CONV#` e `PROFILE`) sobre a tabela `sdr-sessions`, mesmo padrão do leitor do u5; U6 **nunca escreve** na tabela de sessões (owner U1) — o estado de cadência vai para tabela dedicada `sdr-followup-state` (item `SK=FOLLOWUP` por lead).
- **Passo vencido = maior dia da cadência já alcançado** (`due_step`): tick perdido se autocorrige (dia 2 perdido dispara no dia 3) e a idempotência fica na `DuplicateGuard` (`last_step >= step` → `already_followed_up`), não no cálculo de cadência — separação limpa entre "o que venceu" e "o que já foi dito".
- **Guarda de resposta**: a conversa é lida com `messages` (Contrato 5); se a última mensagem do lead (`role=lead`, maior `at` — schema real do produtor U1: `{"role", "text", "at"}`, ver handler.py do conversation-router; `ts` mantido só como fallback de compatibilidade) é posterior ao `last_followup_at` registrado, a cadência para (`lead_replied`) — o lead voltou a conversar, follow-up automático não deve remarcar.
- **Drop explícito** (constraint da unidade): `invalid_created_at`, `no_context` (sem mensagens e sem intenção), `no_channel` (perfil sem `telegram_user_id`) e `cadence_exhausted` (delta além do último dia sem cadência concluída) — cada um com `log_event` dedicado e contador próprio no resumo; cadência concluída vira neutro (`not_due`), não drop.
- **PII-safe em duas camadas**: o builder só recebe `{step, intent, state}` (nunca mensagens brutas nem contato) e os logs estruturados só carregam identificadores (`lead_id`, `step`, contadores) — verificado por testes com PII sintética no `caplog`.
- **Falha não grava estado**: erro de `sendMessage` (GatewayError) ou de escrita do estado incrementa `errors` e deixa o passo pendente — o próximo tick repete (retry assíncrono da Lambda, NFR4).
- **Janela de silêncio global por tick**: fora da janela nenhum lead é contactado e todos os vencidos viram `deferred` (nenhum estado gravado) — o próximo tick dentro da janela retoma; boundaries testados com clock injetável (start inclusivo, end exclusivo, meia-noite, 24h).

## Test coverage summary

- Comando: `COVERAGE_FILE=/tmp/opencode/.cov-fix-u6 .venv/bin/python -m pytest apps/followup/tests --cov=apps/followup --cov-report=term-missing --cov-fail-under=80 -q`
- Resultado (após a rodada de fix): **87 coletados — 87 passed**, **TOTAL 98.52%** (piso 80% atendido, nenhuma meta relaxada).
- Componentes: `handler.py` 100%, `cadence.py` 100%, `silence_window.py` 100%, `duplicate_guard.py` 100%, `telegram_gateway.py` 100%, `structured_log.py` 100%, `conversation_store.py` 96%, `followup.py` 95%, `followup_state.py` 94%, `message_builder.py` 91%.
- Suítes irmãs re-verificadas após a rodada de fix: u1 = 116 passed, u2 = 75 passed, u3 = 71 passed, u4 = 117 passed, u5 = 84 passed + 3 skipped.

## Deviations from the plan

- **Wait states do Step Functions simulados na POC**: a máquina de estados com `Wait` (dia 2/5/9) foi materializada como `CadenceCalculator` injetável + tick diário do EventBridge que resolve o passo vencido por `delta >= dia`, com estado persistido em `sdr-followup-state` (`last_step`/`next_step`/`done`). A migração para Step Functions real (estados `Wait` até timestamp) não muda o Protocol do orquestrador — o calculator é trocado pelo input da execution.
- **Guarda de duplicidade injetável** (`DuplicateGuard`): a restrição "lead que respondeu não deve ser remensurado" não tinha mecanismo definido no Contrato 5; implementado comparando o `ts` da última mensagem do lead (lida da conversa) com `last_followup_at` do estado de cadência. POC: a detecção de resposta é passiva (feita no tick, não em tempo real) — o router continua dono do fluxo ativo.
- **Drop de lead sem canal** (`no_channel`): o Contrato 5 não define o comportamento quando o Lead não tem `telegram_user_id` (leads de portal do u4 têm pseudo-id derivado por hash); U6 trata como drop explícito com log, coerente com a constraint de drop da unidade.
- **Janela de silêncio global por tick** (não por lead): a verificação é feita uma vez por execução; leads vencidos fora da janela viram `deferred` e são retomados no próximo tick dentro da janela. Configuração via `SILENCE_WINDOW_START`/`SILENCE_WINDOW_END` (horas do fuso local — padrão `America/Sao_Paulo`, override via env `TIMEZONE` — padrão 8–18, fim exclusivo, suporta meia-noite e 24h).
- **boto3/requests ausentes no `.venv`**: o wiring do `handler()` é testado com módulos fake injetados em `sys.modules` (padrão já usado pelo u4/u5); dependências ficam declaradas no `requirements.txt` para o build de produção.
- IaC (Terraform — regra do EventBridge, tabela `sdr-followup-state`, agendador do Step Functions) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.

## Deviations da rodada de fix (iteração 2 do gate)

Consumação dos findings de `.aidlc-reviews/code-generation/units/u6-followup/45d170221e303cc2/1.json` — a unidade se ALINHA à u1 (correção de bugs, sem reverter o produtor):

- **R-01 (Critical)** — `_last_lead_message_at` agora lê `at` (schema real da U1: `{"role", "text", "at"}`, ver `apps/conversation-router/handler.py`), mantendo `ts` como fallback de compatibilidade; todos os fixtures de teste migrados para `at` e dois testes novos provam a supressão `lead_replied` contra o shape da U1 (unit + método estático).
- **R-02 (Major)** — processamento isolado por lead: `try/except` em torno de `_process_lead` no loop de `run()` (falha → contador `errors` + `log_event("followup_lead_failed")` PII-safe, tick continua); teste prova que store falhando para um lead não impede os demais.
- **R-03 (Major)** — `GatewayError` não incorpora mais `str(exc)`: mensagem carrega apenas o tipo da exceção (`sendMessage failed: <TipoExceção>`) ou o status HTTP — URL/query/token do bot nunca vazam para o log (`followup_send_failed`); teste de regressão com exceção contendo URL+token.
- **R-04 (Minor)** — NFR4.1 reclassificado como **Deferred** na `traceability.json`: o que está implementado aqui é o retry no próximo tick; a DLQ do alvo EventBridge é config de IaC (estágios infrastructure/deployment — Build and Test).
- **R-05 (Minor)** — docstring do gateway e Step 5 do plan corrigidos: envio é Telegram Bot API `sendMessage` (mesmo mecanismo da U1); Contrato 1 reservado ao webhook de entrada.
- **R-06 (Minor)** — janela de silêncio em horas do fuso LOCAL, padrão `America/Sao_Paulo` (8–18 locais), override via env `TIMEZONE`; testes de boundary em BRT (11–21 UTC) e de override/rejeição de fuso inválido.
- **R-07 (Minor)** — padronização completa em `log_event` JSON (NFR5.1): helper compartilhado `infra/structured_log.py` adotado por `service/followup.py`, `infra/conversation_store.py` e `infra/followup_state.py` — nenhum `logger.warning`/`logger.error` em prosa resta na unidade; testes verificam parse JSON no `caplog`.
