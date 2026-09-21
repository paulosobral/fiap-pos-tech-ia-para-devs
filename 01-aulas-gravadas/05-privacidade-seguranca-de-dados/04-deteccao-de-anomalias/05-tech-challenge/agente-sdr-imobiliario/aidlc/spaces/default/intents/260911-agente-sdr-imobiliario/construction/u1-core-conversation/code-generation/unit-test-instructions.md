# Unit Test Instructions — u1-core-conversation

> Estratégia: `standard` · Metodologia: `test-after` · Escopo: `feature` (piso 80% linhas).

## Setup do framework

- Runner: `pytest` + `pytest-cov` (config em `pyproject.toml`, rootdir do projeto).
- Dependências de teste: `requirements-dev.txt`.

## Comando desta unidade (escopo exato)

```bash
python -m pytest apps/conversation-router/tests/unit/test_pii_masker.py apps/conversation-router/tests/unit/test_guardrails.py apps/conversation-router/tests/unit/test_lead_qualifier.py apps/conversation-router/tests/unit/test_session_store.py apps/conversation-router/tests/unit/test_sales_flow.py apps/conversation-router/tests/unit/test_entities.py apps/conversation-router/tests/integration/test_conversation_router.py --cov=apps/conversation-router --cov-report=term-missing --cov-fail-under=80
```

Comando proibido aqui (seria executado no Build and Test): `pytest` sem filtro de unidade.

## Metas de cobertura

- Piso: 80% de linhas em `apps/conversation-router` (obrigação do escopo `feature`; não relaxável).

## Contagem por componente (standard: 5-8)

- `test_pii_masker.py` — 8 testes
- `test_guardrails.py` — 5 testes
- `test_lead_qualifier.py` — 8 testes
- `test_entities.py` — 8 testes
- `test_session_store.py` — 4 testes
- `test_sales_flow.py` — 8 testes
- `test_conversation_router.py` — 9 testes

## Mocking/stubbing

- `boto3`: `moto`-free — doubles simples via `unittest.mock.MagicMock` (Cliente DynamoDB e SQS stubados).
- LLM (OpenRouter/LiteLLM): stub de função (`llm_classify_intent` injetável).
- Telegram/SQS: handlers retornam dicts; clientes injetados por parâmetro (injeção de dependência, sem singletons).

## Dados de teste

- Payloads Telegram sintéticos em fixtures (`apps/conversation-router/tests/integration/fixtures.py`).
- PII de teste: "João Silva", "joao@empresa.com", "+55 11 91234-5678", "12.345.678/0001-95".
