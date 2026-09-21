# Unit Test Instructions — u3-async-crm

## Framework e configuração

- pytest + pytest-cov já presentes no `.venv` da raiz do repo (`pytest>=8.0`, `pytest-cov>=5.0` em `requirements-dev.txt`).
- O `pyproject.toml` da raiz **não foi alterado** (pythonpath/testpaths continuam apontando para `apps/conversation-router`). O bootstrap de path desta unidade é feito por `apps/crm-adapter/tests/conftest.py`, que insere `apps/crm-adapter` na frente do `sys.path` — os imports `handler`, `service.*` e `infra.*` resolvem para esta unidade (mesmo estilo flat do u1/u2).
- **SDK MCP não é exigido** para a suíte: o import é protegido em `service/crm_gateway.py` (try/except ImportError no nível de módulo) e os testes do `McpCrmGateway` usam cliente MCP fake injetado (MagicMock).
- O CRM simulado é 100% local: `CsvCrmGateway` grava em arquivo CSV temporário (`tmp_path` do pytest) — nenhum serviço externo é necessário.
- boto3/requests não são exigidos: wiring de produção (`handler()`) usa `monkeypatch.setitem(sys.modules, ...)` nos testes de integração.

## Como rodar ESTA unidade (comando exato, da raiz do repo)

```bash
COVERAGE_FILE=/tmp/.cov-u3 .venv/bin/python -m pytest apps/crm-adapter/tests --cov=apps/crm-adapter --cov-report=term-missing --cov-fail-under=80 -q
```

- Escopo: apenas `apps/crm-adapter/tests` (117 testes no momento da rodada de fix; comando jamais dispara as suítes do u1/u2 — rodá-las juntas quebra o bootstrap de path, cada unidade tem comando próprio). `COVERAGE_FILE` isolado evita colisão com as suítes paralelas.
- Comando de compilação de validação: `.venv/bin/python -m compileall apps/crm-adapter`.

## Cobertura esperada

- Piso obrigatório (Testing Contract, escopo `feature`): **80% de linhas** sobre `apps/crm-adapter` — `--cov-fail-under=80` no comando.
- Resultado atual (pós-fix): **99.59%** (TOTAL: 1224 stmts, 5 miss). Misses restantes são linhas defensivas de arquivos de teste (ramos `else: raise AssertionError` e fallback de `open` em teste de I/O).

## Mocking/stubbing

- Doubles: `unittest.mock.MagicMock` (padrão do u1/u2) para CRM gateway, flow callback, DynamoDB e http client.
- Orquestrador (`CrmAdapter`): `crm`/`status`/`flow`/`sessions` injetados como mocks; erros simulados via `side_effect` (`CrmError` → retry, `FlowError` → retry, `RuntimeError` inesperado → retry).
- CRM simulado: `CsvCrmGateway` contra `CsvStore` em `tmp_path`; erros de I/O injetados via `monkeypatch` (`os.replace`, `builtins.open`, `store.save`).
- MCP: cliente fake com `call_tool(name, arguments)`; disponibilidade do SDK simulada com `monkeypatch.setattr(crm_gateway_module, "_HAS_MCP", ...)`.
- Wiring de produção (`handler()`): `monkeypatch.setitem(sys.modules, "boto3"/"requests", MagicMock())` + `monkeypatch.setenv(...)` nos testes de integração.

## Gestão de dados de teste

- Mensagens do Contract 4 geradas por helpers (`crm_message()`, `sqs_record()`); corpo SQS serializado com `json.dumps`.
- Nenhuma credencial real: tokens de teste são literais inertes (`"sec"`, `"tok"`); env só via monkeypatch (nunca hardcoded em código de produção).
- `lead_data` dos fixtures carrega PII sintética (nome/e-mail/telefone fictícios) — usada também para provar que os logs estruturados não vazam PII, inclusive no caminho `invalid_body` (preview mascarado com o padrão oficial) e nos caminhos de erro (JSON via `log_event`); assertions sobre `caplog.text`.
- Snapshot do payload REAL do produtor (`real_producer_message()`, espelho do `_enqueue_crm` da u1): casos sem registro de PII (email/phone null, fallback de `name`) e com registro completo.
- Arquivos CSV de teste vivem em `tmp_path` e são descartados pelo pytest.
