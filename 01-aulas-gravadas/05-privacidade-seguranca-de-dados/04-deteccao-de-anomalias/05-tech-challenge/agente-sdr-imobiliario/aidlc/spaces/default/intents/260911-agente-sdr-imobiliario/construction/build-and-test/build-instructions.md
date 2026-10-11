# Build Instructions — Agente SDR Imobiliário (POC)

## Pré-requisitos
- Python 3.11+ (o projeto usa `.venv/` na raiz)
- Sem dependências de infraestrutura externa para o build e para as suítes unitárias/integração (DynamoDB/SQS/EventBridge/Telegram/KMS são fakes/injeção de dependência)
- `requirements-dev.txt` instala também `scikit-learn` (SklearnScorer do u5 roda de verdade), `streamlit` (testes de render e `AppTest` do dashboard-ui) e `Pillow` (filtro de fotos do crawler e favicon)
- `podman` e `terraform` só para o deploy (`start.sh`), não para o build de testes

## Instalação de dependências
```bash
# A partir da raiz do projeto (agente-sdr-imobiliario/)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"   # ou: .venv/bin/pip install -r requirements-dev.txt
```

## Setup de ambiente
- Variáveis de ambiente **não são exigidas** para as suítes unitárias/integração (injeção de dependência; sem rede).
- O **gate de qualidade com LLM real** (`apps/conversation-router/tests/quality`) exige `LLM_API_KEY` ou `OPENROUTER_API_KEY` (carregue `secrets.local.env`); sem chave ele é pulado. Custa crédito de LLM e leva ~7 min.
- Cobertura isolada: exporte `COVERAGE_FILE=/tmp/.cov-<unit>` antes de rodar suítes com `--cov` para evitar poluir `.coverage` na raiz.

## Build
O "build" deste POC Python é a compilação a byte-code + smoke de import (via suítes):

```bash
.venv/bin/python -m compileall -q apps    # exit 0 = build OK
```

- `compileall` valida sintaxe de todos os pacotes (`apps/{conversation-router,voice-adapter,crm-adapter,contact-ingest,anomaly-detector,followup,dashboard-api,dashboard-ui}`).
- O smoke de import real acontece ao executar as suítes: cada `apps/*/tests/conftest.py` insere o diretório do app em `sys.path` e os módulos do handler são importados durante os testes.

## Empacotamento das Lambdas (feito pelo `start.sh`, fase 4)
- `pip install -r apps/<app>/requirements.txt -t dist/<app>/ --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:` + `zip`.
- O `crm-adapter` mantém as pastas `*.dist-info` no pacote: `httpx2`/`mcp` leem a própria versão via `importlib.metadata` ao importar (sem elas a Lambda quebra no import). Pacote ~28 MB compactado (limite de upload 50 MB).
- Imagens ECS (router, voice-adapter, dashboard-ui): `podman build`/`push` no ECR. O `Dockerfile` do dashboard-ui copia `assets/` e `.streamlit/` (logo e tema).

## Verificação do build
```bash
COVERAGE_FILE=/tmp/.cov-<unit> .venv/bin/python -m pytest apps/<app>/tests \
  --cov=apps/<app> --cov-report=term --cov-fail-under=80 -q
```
Cobertura mínima contratada: **80% por unit** (`dashboard-ui` agora roda com streamlit real: 80,32%, no limite do piso).

Gate com LLM real (antes de subir; o `start.sh` executa com retry `--lf` dos que falharem):
```bash
set -a; source secrets.local.env; set +a
.venv/bin/python -m pytest apps/conversation-router/tests/quality -q
```

## Troubleshooting
| Problema | Causa | Solução |
|---|---|---|
| `ModuleNotFoundError: apps.<app>` | Nomes de app contêm hífen; `pyproject.toml` só registra `pythonpath` de `conversation-router` | Rode pytest apontando `apps/<app>/tests` — o `conftest.py` do app resolve o `sys.path` |
| `.coverage` aparece na raiz | Comando rodado sem `COVERAGE_FILE` isolado | Exporte `COVERAGE_FILE=/tmp/.cov-<unit>` (e remova `.coverage` órfão) |
| Fingerprint do engine AI-DLC reclama de source changed | `__pycache__`/`.pytest_cache`/`.coverage` alterados | `find apps -type d -name __pycache__ -exec rm -rf {} +; rm -rf .pytest_cache .coverage` |
| `skipped` nos testes do dashboard-ui | `streamlit` ausente do venv (`importorskip`) | `pip install -r requirements-dev.txt`; com streamlit instalado nada é pulado |
| Gate de qualidade falha com `429` | Provedor upstream do modelo primário limitado | O cliente cai para o modelo fallback; rodar de novo (`--lf`) |
| `ImportError` ao rodar `scripts/hubspot_authorize.py` | SDK `mcp` novo renomeou `streamablehttp_client` | Usar a versão do repo (`streamable_http_client`) |
