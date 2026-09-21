# Unit Test Instructions — u2-async-voice

## Framework e configuração

- pytest + pytest-cov já presentes no `.venv` da raiz do repo (`pytest>=8.0`, `pytest-cov>=5.0` em `requirements-dev.txt`).
- O `pyproject.toml` da raiz **não foi alterado** (pythonpath/testpaths continuam apontando para `apps/conversation-router`). O bootstrap de path desta unidade é feito por `apps/voice-adapter/tests/conftest.py`, que insere `apps/voice-adapter` na frente do `sys.path` — os imports `handler`, `service.*` e `infra.*` resolvem para esta unidade (mesmo estilo flat do u1).
- **faster-whisper não é exigido** para a suíte: o import é protegido em `service/transcriber.py` (try/except ImportError no nível de módulo) e os testes usam `model_factory` injetável/stubs (MagicMock).
- ffmpeg não é exigido: `subprocess.run` é mockado via monkeypatch nos testes do transcriber.

## Como rodar ESTA unidade (comando exato, da raiz do repo)

```bash
.venv/bin/python -m pytest apps/voice-adapter/tests --cov=apps/voice-adapter --cov-report=term-missing --cov-fail-under=80
```

- Escopo: apenas `apps/voice-adapter/tests` (58 testes no momento da gravação; comando jamais dispara a suíte do u1).
- Comando de compilação de validação: `.venv/bin/python -m compileall apps/voice-adapter`.

## Cobertura esperada

- Piso obrigatório (Testing Contract, escopo `feature`): **80% de linhas** sobre `apps/voice-adapter` — `--cov-fail-under=80` no comando.
- Resultado atual: **99.86%**. Única linha não coberta: `service/transcriber.py:15` (import protegido `from faster_whisper import WhisperModel` — só executa com o pacote pesado instalado).

## Mocking/stubbing

- Doubles: `unittest.mock.MagicMock` (padrão do u1) para http client, DynamoDB, transcriber, router e telegram.
- Transcriber: interface `Transcriber` (Protocol); testes de orquestrador injetam `MagicMock`; testes do `WhisperTranscriber` injetam `model_factory` real-fake e mockam `service.transcriber.subprocess.run` (ffmpeg). Disponibilidade do faster-whisper simulada com `monkeypatch.setattr(transcriber_module, "_HAS_FASTER_WHISPER"/"_WhisperModel", ...)`.
- Wiring de produção (`handler()`): `monkeypatch.setitem(sys.modules, "boto3"/"requests", MagicMock())` + `monkeypatch.setenv(...)` nos testes de integração.

## Gestão de dados de teste

- Mensagens do Contract 3 geradas por helpers (`voice_message()`, `sqs_record()`); corpo SQS serializado com `json.dumps`.
- Nenhuma credencial real: tokens de teste são literais inertes (`"tok"`, `"sec"`); env só via monkeypatch (nunca hardcoded em código de produção).
- Áudio é `bytes` fictício (`b"ogg-bytes"`); nenhum arquivo de áudio real na suíte.
