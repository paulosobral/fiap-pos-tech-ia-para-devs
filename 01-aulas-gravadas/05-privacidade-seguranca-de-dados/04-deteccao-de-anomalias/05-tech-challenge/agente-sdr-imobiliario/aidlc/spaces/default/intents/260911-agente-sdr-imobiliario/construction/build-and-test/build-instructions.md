# Build Instructions — Agente SDR Imobiliário (POC)

## Pré-requisitos
- Python 3.11+ (o projeto usa `.venv/` na raiz)
- Sem dependências de infraestrutura externa para o build (DynamoDB/SQS/EventBridge/Telegram são simulados por fakes/injeção de dependência nos testes)

## Instalação de dependências
```bash
# A partir da raiz do projeto (agente-sdr-imobiliario/)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # ou: .venv/bin/pip install -r requirements-dev.txt
```

## Setup de ambiente
- Variáveis de ambiente **não são exigidas** para build/testes (handlers usam injeção de dependência; nenhum teste faz chamada de rede).
- Cobertura isolada: exporte `COVERAGE_FILE=/tmp/.cov-<unit>` antes de rodar suítes com `--cov` para evitar poluir `.coverage` na raiz.

## Build
O "build" deste POC Python é a compilação a byte-code + smoke de import (via suítes):

```bash
.venv/bin/python -m compileall -q apps    # exit 0 = build OK
```

- `compileall` valida sintaxe de todos os pacotes (`apps/{conversation-router,voice-adapter,crm-adapter,contact-ingest,anomaly-detector,followup,dashboard-api,dashboard-ui}`).
- O smoke de import real acontece ao executar as suítes: cada `apps/*/tests/conftest.py` insere o diretório do app em `sys.path` e os módulos do handler são importados durante os testes.

## Verificação do build
```bash
COVERAGE_FILE=/tmp/.cov-<unit> .venv/bin/python -m pytest apps/<app>/tests \
  --cov=apps/<app> --cov-report=term --cov-fail-under=80 -q
```
Cobertura mínima contratada: **80% por unit** (`u7`: piso sobre `dashboard-api`; `dashboard-ui` é smoke).

## Troubleshooting
| Problema | Causa | Solução |
|---|---|---|
| `ModuleNotFoundError: apps.<app>` | Nomes de app contêm hífen; `pyproject.toml` só registra `pythonpath` de `conversation-router` | Rode pytest apontando `apps/<app>/tests` — o `conftest.py` do app resolve o `sys.path` |
| `.coverage` aparece na raiz | Comando rodado sem `COVERAGE_FILE` isolado | Exporte `COVERAGE_FILE=/tmp/.cov-<unit>` (e remova `.coverage` órfão) |
| Fingerprint do engine AI-DLC reclama de source changed | `__pycache__`/`.pytest_cache`/`.coverage` alterados | `find apps -type d -name __pycache__ -exec rm -rf {} +; rm -rf .pytest_cache .coverage` |
| `skipped` nos testes do dashboard-ui | Testes smoke pulados por ausência de browser/node | Aceito pelo contrato: UI é smoke test |
