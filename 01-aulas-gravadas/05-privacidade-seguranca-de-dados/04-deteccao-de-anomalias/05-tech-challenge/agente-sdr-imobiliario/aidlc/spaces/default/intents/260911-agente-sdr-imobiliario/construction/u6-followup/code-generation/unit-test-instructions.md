# Unit Test Instructions — u6-followup

## Framework e configuração

- pytest + pytest-cov já presentes no `.venv` da raiz do repo (`pytest>=8.0`, `pytest-cov>=5.0` em `requirements-dev.txt`).
- O `pyproject.toml` da raiz **não foi alterado** (pythonpath/testpaths continuam apontando para `apps/conversation-router`). O bootstrap de path desta unidade é feito por `apps/followup/tests/conftest.py`, que insere `apps/followup` na frente do `sys.path` — os imports `handler`, `service.*` e `infra.*` resolvem para esta unidade (mesmo estilo flat do u1–u5).
- **Nenhum serviço AWS é exigido**: o evento EventBridge é JSON puro; DynamoDB entra por DI (fakes in-memory/MagicMock). boto3 e requests não são exigidos pela suíte: o wiring de produção (`handler()`) usa `monkeypatch.setitem(sys.modules, ...)` com módulos fake (mesmo padrão do u4/u5).
- **Tempo é determinístico**: o clock é injetável em toda a cadeia (`CadenceCalculator.now_fn`, `SilenceWindow.now_fn`, `FollowupService.now_fn`). Os testes de unidade fixam `FIXED_NOW = 2026-09-20T12:00Z` (meio-dia, dentro da janela padrão 8–18 UTC); os testes de integração derivam `created_at` do relógio real e forçam a janela via env (`SILENCE_WINDOW_START=0`/`END=0` = janela 24h).

## Como rodar ESTA unidade (comando exato, da raiz do repo)

```bash
.venv/bin/python -m pytest apps/followup/tests --cov=apps/followup --cov-report=term-missing --cov-fail-under=80
```

- Escopo: apenas `apps/followup/tests` (comando jamais dispara as suítes do u1/u2/u3/u4/u5 — rodá-las juntas quebra o bootstrap de path, cada unidade tem comando próprio).
- Comando de compilação de validação: `.venv/bin/python -m compileall apps/followup`.

## Cobertura esperada

- **77 testes** (69 unitários + 8 de integração), todos passando; **TOTAL 98%** de linhas (piso 80% contratado — nenhuma meta relaxada).
- Componentes: `handler.py` 100%, `cadence.py` 100%, `silence_window.py` 100%, `duplicate_guard.py` 100%, `telegram_gateway.py` 100%, `followup_state.py` 95%, `conversation_store.py` 96%, `followup.py` 93%, `message_builder.py` 91%.
- Boundaries obrigatórios cobertos: bordas da janela (start inclusivo, end exclusivo, janela que cruza meia-noite, janela 24h), passos dia 2/5/9 + dia perdido que se autocorrige + expiração, guarda de passo repetido (idempotência do tick), guarda de lead que respondeu, drop explícito sem contexto/canal/created_at inválido, falha de envio sem gravação de estado (retry no próximo tick).

## Mocking/stubbing

- Fakes in-memory por componente (`FakeConversations`, `FakeState`, `FakeGuard`, `FakeTelegram`, `FakeScanClient`, `FakeStateClient`, `FakeDynamo`, `FakeHttp`), injetados por construtor — zero rede/zero AWS.
- `requests` e `boto3` nunca importados de verdade na suíte: o handler recebe módulos fake em `sys.modules` e o `FakeHttp` grava as chamadas para assertions de URL/payload do `sendMessage`.
- Logs estruturados verificados com `caplog` (assertions de ausência de PII em `caplog.text`).

## Gestão de dados de teste

- Fixtures determinísticas: lead criado há 2/5/9/10 dias (`DAY_2`, `DAY_5`, `DAY_9`, `DAY_10`), perfil com `telegram_user_id` e `intent`, conversa com mensagens `{role, text, ts}`.
- PII sintética (e-mail/telefone/bairro em `messages`) usada para provar que os logs estruturados não vazam PII; o builder prova PII-safety por construção (só recebe `step`/`intent`/`state`).
- Nenhuma credencial real; env só via `monkeypatch.setenv` (`SESSIONS_TABLE`, `FOLLOWUP_TABLE`, `TELEGRAM_BOT_TOKEN`, `FOLLOWUP_CADENCE_DAYS`, `SILENCE_WINDOW_START`, `SILENCE_WINDOW_END`) — nada hardcoded em código de produção.
- Sem arquivos em disco: toda a suíte é in-memory.
