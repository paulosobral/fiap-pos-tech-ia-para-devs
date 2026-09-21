# Code Summary — u6-followup

## Files created/modified

Todos os arquivos são novos, sob `apps/followup/` (nenhum app existente foi tocado):

| Arquivo | Papel |
|---------|-------|
| `apps/followup/handler.py` | Entry point Lambda — tick diário do EventBridge; wiring DI completo (ConversationStore, FollowupStateStore, DuplicateGuard, CadenceCalculator, SilenceWindow, FollowupMessageBuilder, TelegramGateway); env: `SESSIONS_TABLE`, `FOLLOWUP_TABLE`, `TELEGRAM_BOT_TOKEN`, `FOLLOWUP_CADENCE_DAYS`, `SILENCE_WINDOW_START/END` |
| `apps/followup/requirements.txt` | Dependências de produção (`boto3>=1.34`, `requests>=2.31`) |
| `apps/followup/infra/conversation_store.py` | Leitor do Contrato 5: scan paginado `begins_with(SK, CONV#)` (conversas) e `begins_with(SK, PROFILE)` (mapa `lead_id → telegram_user_id/intent`); unmarshal de tipos/JSON; itens sem identificadores ignorados com log; erros → `ConversationStoreError` |
| `apps/followup/infra/followup_state.py` | Estado próprio de cadência (tabela `sdr-followup-state`, item `PK=LEAD#<id>, SK=FOLLOWUP`): `last_step`, `last_followup_at`, `next_step`, `done`; erros → `FollowupStateError` |
| `apps/followup/infra/silence_window.py` | FR8.3 — janela de silêncio configurável (horas UTC, fim exclusivo), suporta janela que cruza meia-noite (`start > end`) e janela 24h (`start == end`); clock injetável; `from_env` |
| `apps/followup/service/cadence.py` | FR8.1 — `CadenceCalculator` (dias 2/5/9 configuráveis, `due_step` = maior dia já alcançado, `is_expired`, `next_step`); `parse_cadence_days`; `parse_iso8601`; clock injetável |
| `apps/followup/service/duplicate_guard.py` | Guarda anti-spam: `already_followed_up` (passo ≤ último enviado — idempotência do tick) e `lead_replied` (mensagem do lead posterior ao último follow-up) |
| `apps/followup/service/followup.py` | Orquestrador: conversa mais recente por lead → drop explícito (invalid_created_at / cadence_exhausted) → passo vencido → janela de silêncio (defer) → guarda → contexto PII-safe → mensagem PT-BR → envio → persistir próximo passo; resumo com contadores e logs estruturados `log_event` |
| `apps/followup/service/message_builder.py` | FR8.2 — mensagens PT-BR por passo (2/5/9) personalizadas pela intenção (compra/locação/investimento); PII-safe por construção (recebe só `step`/`intent`/`state`) |
| `apps/followup/service/telegram_gateway.py` | Contrato 1 — `sendMessage` via http injetado (padrão voice-adapter); falha → `GatewayError` propagada (estado não gravado, retry no próximo tick) |
| `apps/followup/tests/conftest.py` | Bootstrap de path (`sys.path` → `apps/followup`), estilo flat u1–u5 |
| `apps/followup/tests/unit/test_*.py` | 7 suítes unitárias (cadence, silence_window, message_builder, telegram_gateway, followup_state, conversation_store, duplicate_guard, followup) |
| `apps/followup/tests/integration/test_followup_pipeline.py` | Pipeline ponta a ponta: handler + FakeDynamo + FakeHttp + env, idempotência, janela fechada, cadência custom, supressão por resposta, PII-safe nos logs, token ausente, payload não-dict |

## Key implementation decisions

- **Contrato 5 com dois scans**: a cadência precisa de `created_at` da conversa e do `telegram_user_id`/`intent` do Lead. U6 é reader do Contrato 5 — a leitura é feita com dois scans paginados (`CONV#` e `PROFILE`) sobre a tabela `sdr-sessions`, mesmo padrão do leitor do u5; U6 **nunca escreve** na tabela de sessões (owner U1) — o estado de cadência vai para tabela dedicada `sdr-followup-state` (item `SK=FOLLOWUP` por lead).
- **Passo vencido = maior dia da cadência já alcançado** (`due_step`): tick perdido se autocorrige (dia 2 perdido dispara no dia 3) e a idempotência fica na `DuplicateGuard` (`last_step >= step` → `already_followed_up`), não no cálculo de cadência — separação limpa entre "o que venceu" e "o que já foi dito".
- **Guarda de resposta**: a conversa é lida com `messages` (Contrato 5); se a última mensagem do lead (`role=lead`, maior `ts`) é posterior ao `last_followup_at` registrado, a cadência para (`lead_replied`) — o lead voltou a conversar, follow-up automático não deve remarcar.
- **Drop explícito** (constraint da unidade): `invalid_created_at`, `no_context` (sem mensagens e sem intenção), `no_channel` (perfil sem `telegram_user_id`) e `cadence_exhausted` (delta além do último dia sem cadência concluída) — cada um com `log_event` dedicado e contador próprio no resumo; cadência concluída vira neutro (`not_due`), não drop.
- **PII-safe em duas camadas**: o builder só recebe `{step, intent, state}` (nunca mensagens brutas nem contato) e os logs estruturados só carregam identificadores (`lead_id`, `step`, contadores) — verificado por testes com PII sintética no `caplog`.
- **Falha não grava estado**: erro de `sendMessage` (GatewayError) ou de escrita do estado incrementa `errors` e deixa o passo pendente — o próximo tick repete (retry assíncrono da Lambda, NFR4).
- **Janela de silêncio global por tick**: fora da janela nenhum lead é contactado e todos os vencidos viram `deferred` (nenhum estado gravado) — o próximo tick dentro da janela retoma; boundaries testados com clock injetável (start inclusivo, end exclusivo, meia-noite, 24h).

## Test coverage summary

- Comando: `.venv/bin/python -m pytest apps/followup/tests --cov=apps/followup --cov-report=term-missing --cov-fail-under=80`
- Resultado: **77 coletados — 77 passed**, **TOTAL 97.87%** (piso 80% atendido, nenhuma meta relaxada).
- Componentes: `handler.py` 100%, `cadence.py` 100%, `silence_window.py` 100%, `duplicate_guard.py` 100%, `telegram_gateway.py` 100%, `conversation_store.py` 96%, `followup_state.py` 95%, `followup.py` 93%, `message_builder.py` 91%.
- Suítes anteriores re-verificadas após a unidade: u1 = 50 passed, u2 = 58 passed, u3 = 86 passed, u4 = 50 passed, u5 = 51 passed + 1 skipped.

## Deviations from the plan

- **Wait states do Step Functions simulados na POC**: a máquina de estados com `Wait` (dia 2/5/9) foi materializada como `CadenceCalculator` injetável + tick diário do EventBridge que resolve o passo vencido por `delta >= dia`, com estado persistido em `sdr-followup-state` (`last_step`/`next_step`/`done`). A migração para Step Functions real (estados `Wait` até timestamp) não muda o Protocol do orquestrador — o calculator é trocado pelo input da execution.
- **Guarda de duplicidade injetável** (`DuplicateGuard`): a restrição "lead que respondeu não deve ser remensurado" não tinha mecanismo definido no Contrato 5; implementado comparando o `ts` da última mensagem do lead (lida da conversa) com `last_followup_at` do estado de cadência. POC: a detecção de resposta é passiva (feita no tick, não em tempo real) — o router continua dono do fluxo ativo.
- **Drop de lead sem canal** (`no_channel`): o Contrato 5 não define o comportamento quando o Lead não tem `telegram_user_id` (leads de portal do u4 têm pseudo-id derivado por hash); U6 trata como drop explícito com log, coerente com a constraint de drop da unidade.
- **Janela de silêncio global por tick** (não por lead): a verificação é feita uma vez por execução; leads vencidos fora da janela viram `deferred` e são retomados no próximo tick dentro da janela. Configuração via `SILENCE_WINDOW_START`/`SILENCE_WINDOW_END` (horas UTC, padrão 8–18, fim exclusivo, suporta meia-noite e 24h).
- **boto3/requests ausentes no `.venv`**: o wiring do `handler()` é testado com módulos fake injetados em `sys.modules` (padrão já usado pelo u4/u5); dependências ficam declaradas no `requirements.txt` para o build de produção.
- IaC (Terraform — regra do EventBridge, tabela `sdr-followup-state`, agendador do Step Functions) não gerado neste estágio — handled pelos estágios de infrastructure/deployment.
