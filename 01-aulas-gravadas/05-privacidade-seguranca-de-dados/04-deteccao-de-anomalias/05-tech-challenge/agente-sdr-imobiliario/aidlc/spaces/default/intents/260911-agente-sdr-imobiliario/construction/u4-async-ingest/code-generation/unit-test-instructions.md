# Unit Test Instructions — u4-async-ingest

## Framework e configuração

- pytest + pytest-cov já presentes no `.venv` da raiz do repo (`pytest>=8.0`, `pytest-cov>=5.0` em `requirements-dev.txt`).
- O `pyproject.toml` da raiz **não foi alterado** (pythonpath/testpaths continuam apontando para `apps/conversation-router`). O bootstrap de path desta unidade é feito por `apps/contact-ingest/tests/conftest.py`, que insere `apps/contact-ingest` na frente do `sys.path` — os imports `handler`, `service.*` e `infra.*` resolvem para esta unidade (mesmo estilo flat do u1/u2/u3).
- **Nenhum serviço AWS é exigido**: o evento SES é JSON puro; DynamoDB e HTTP entram por DI (clientes fake/MagicMock). boto3/requests não são exigidos pela suíte: o wiring de produção (`handler()`) usa `monkeypatch.setitem(sys.modules, ...)` nos testes de integração.
- O parser de e-mail usa apenas stdlib (`email`, `base64`, `re`) — sem dependência extra.

## Como rodar ESTA unidade (comando exato, da raiz do repo)

```bash
.venv/bin/python -m pytest apps/contact-ingest/tests --cov=apps/contact-ingest --cov-report=term-missing --cov-fail-under=80
```

- Escopo: apenas `apps/contact-ingest/tests` (comando jamais dispara as suítes do u1/u2/u3 — rodá-las juntas quebra o bootstrap de path, cada unidade tem comando próprio).
- Comando de compilação de validação: `.venv/bin/python -m compileall apps/contact-ingest`.

## Cobertura esperada

- Piso obrigatório (Testing Contract, escopo `feature`): **80% de linhas** sobre `apps/contact-ingest` — `--cov-fail-under=80` no comando.
- Meta da unidade: ≥ 95% (mesmo patamar do u2/u3); misses aceitáveis apenas em ramos defensivos dentro de arquivos de teste.

## Mocking/stubbing

- Doubles: `unittest.mock.MagicMock` (padrão do u1/u2/u3) para parser, dedupe, sessões, router, cliente DynamoDB e http client.
- Dedupe: cliente fake com namespace `exceptions.ConditionalCheckFailedException` real (type criado no teste) para exercitar o put condicional; duplicata simulada via `side_effect`.
- Orquestrador (`ContactIngest`): `parser`/`dedupe`/`sessions`/`router` injetados; erros simulados via `side_effect` (`DedupeError` → retry, `SessionError` → retry + rollback do dedupe, `RouterError` → retry + rollback, exceção inesperada → retry sem derrubar o evento).
- Parser: `HeuristicEmailParser` real contra fixtures de evento SES (`ses_record()`), incluindo corpo MIME cru em base64.
- Wiring de produção (`handler()`): `monkeypatch.setitem(sys.modules, "boto3"/"requests", MagicMock())` + `monkeypatch.setenv("ROUTER_BASE_URL", ...)`; retry pendente simulado fazendo `post` falhar (assert `PendingRetryError`).

## Gestão de dados de teste

- Eventos SES gerados por helpers (`ses_record()`, `ses_event()`): `commonHeaders` (from/subject) e `content` base64 opcional (MIME cru).
- Corpos de portal em pt-BR com campos rotulados (`Nome:`, `E-mail:`, `Telefone:`) e variantes de formato (com/sem DDD, MIME multipart).
- Nenhuma credencial real: tokens de teste são literais inertes (`"sec"`); env só via monkeypatch (nunca hardcoded em código de produção).
- Fixtures carregam PII sintética (nome/e-mail/telefone fictícios) — usada também para provar que os logs estruturados não vazam PII (assertions sobre `caplog.text`).
- Sem arquivos em disco: toda a suíte é in-memory (nada em `tmp_path` é obrigatório).
