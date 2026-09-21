# Unit Test Instructions — u5-anomaly

## Framework e configuração

- pytest + pytest-cov já presentes no `.venv` da raiz do repo (`pytest>=8.0`, `pytest-cov>=5.0` em `requirements-dev.txt`).
- O `pyproject.toml` da raiz **não foi alterado** (pythonpath/testpaths continuam apontando para `apps/conversation-router`). O bootstrap de path desta unidade é feito por `apps/anomaly-detector/tests/conftest.py`, que insere `apps/anomaly-detector` na frente do `sys.path` — os imports `handler`, `service.*` e `infra.*` resolvem para esta unidade (mesmo estilo flat do u1/u2/u3/u4).
- **Nenhum serviço AWS é exigido**: o evento EventBridge é JSON puro; DynamoDB entra por DI (fakes in-memory/MagicMock). boto3 não é exigido pela suíte: o wiring de produção (`handler()`) usa `monkeypatch.setitem(sys.modules, ...)` com módulo fake (mesmo padrão do u4).
- **scikit-learn não é exigido pela suíte**: o `.venv` não possui scikit-learn/numpy (verificado). Os testes exercitam o `HeuristicScorer` (stdlib) ponta a ponta; as guardas de dependência do `SklearnScorer` são testadas escondendo os módulos via `monkeypatch`, e o ensemble real usa `pytest.importorskip("sklearn")` (pula automaticamente onde sklearn não existe).

## Como rodar ESTA unidade (comando exato, da raiz do repo)

```bash
.venv/bin/python -m pytest apps/anomaly-detector/tests --cov=apps/anomaly-detector --cov-report=term-missing --cov-fail-under=80
```

- Escopo: apenas `apps/anomaly-detector/tests` (comando jamais dispara as suítes do u1/u2/u3/u4 — rodá-las juntas quebra o bootstrap de path, cada unidade tem comando próprio).
- Comando de compilação de validação: `.venv/bin/python -m compileall apps/anomaly-detector`.

## Cobertura esperada

- Piso contratado: 80% de linhas sobre `apps/anomaly-detector` (`--cov-fail-under=80`).
- Obtido na execução: **94.53% TOTAL** (52 testes coletados: 51 passed, 1 skipped).
- `service/scorer.py` fica abaixo do TOTAL (70%): as linhas do ensemble sklearn real só executam com scikit-learn instalado (teste com `importorskip` roda em CI que tiver a dependência). Os caminhos executáveis sem sklearn — guarda de dependência ausente, fallback por lote pequeno e heurística completa — estão 100% cobertos. Nenhum piso foi relaxado: a meta contratada é sobre a unidade, e ela é atendida com folga.

## Mocking/stubbing

- DynamoDB: `FakeDynamo` in-memory (scan com filtro `CONV#`, put/update/query de alertas) e `MagicMock` com `side_effect` para falhas — nunca AWS real.
- Clock injetável: o orquestrador recebe `now_fn` fixo (`FIXED_NOW = 2026-09-20T03:30Z`); nos testes de `handler()`, `handler.utc_now` é patchado para o mesmo instante — features e `detected_at` determinísticos.
- Dependência ML ausente: `monkeypatch.setitem(sys.modules, "sklearn...": None)` provoca o `ImportError` real do import guardado, validando a mensagem de erro do `ScorerDependencyError`.
- boto3 ausente no `.venv`: módulo fake `boto3` injetado em `sys.modules` retorna o mesmo `FakeDynamo` para ambos os stores do wiring.

## Gestão de dados de teste

- Fixtures determinísticas: conversa normal (1 mensagem banal às 10:00 UTC, score 0.0) vs conversa suspeita (30 mensagens × ~1400 chars com palavras negativas às 03:00 UTC, score 0.775 ≥ 0.7).
- PII sintética (e-mail/telefone/texto com palavra-gatilho) usada para provar que os logs estruturados não vazam PII (assertions sobre `caplog.text`).
- Nenhuma credencial real; env só via `monkeypatch.setenv` (`SESSIONS_TABLE`, `ALERTS_TABLE`, `ANOMALY_SCORER`, `ANOMALY_THRESHOLD`) — nada hardcoded em código de produção.
- Sem arquivos em disco: toda a suíte é in-memory.
