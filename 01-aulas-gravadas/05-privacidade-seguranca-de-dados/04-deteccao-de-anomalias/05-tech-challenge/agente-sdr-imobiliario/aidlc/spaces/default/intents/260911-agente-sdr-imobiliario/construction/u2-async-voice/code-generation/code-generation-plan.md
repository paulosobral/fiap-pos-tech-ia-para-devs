# Code Generation Plan — u2-async-voice

> Unidade U2 (Async Voice). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: contract-summary.md (Contratos 3 e 5), unit-of-work.md, requirements.md, referência de convenções: u1-core-conversation.

## Testing Contract

```json
{
  "version": 1,
  "methodology": "test-after",
  "source": "team",
  "ordering": "Implementamos cada camada testável aplicável e, em seguida, escrevemos e executamos os testes dessa camada antes de avançar para a próxima.",
  "scope": "feature",
  "test_strategy": "standard",
  "project_type": "greenfield",
  "contract_sha256": "sha256:f1f75c7c9423814cb8f14b07dbb425fb0ba3324465c44388f61d1088141f6048"
}
```

Obrigações: 5-8 testes por componente; testes de unidade + integração para boundaries chave; piso de cobertura de 80% de linhas; execução em CI antes do merge. Nenhuma meta pode ser relaxada para fazer uma etapa passar.

## Steps

- [x] **Step 1** — Estrutura do projeto e configuração de produção (layout `apps/voice-adapter/{handler.py, service/, infra/, tests/, requirements.txt}` conforme PRD §7.4). *(História: setup da unidade U2)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov já no `.venv` da raiz; path bootstrap via `tests/conftest.py`, pyproject da raiz intocado) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados/segurança: `infra/session_store.py` SessionLookup (leitor do Contract 5: resolve `lead_id` pela GSI `telegram-user-index` e consulta a conversa pela chave composta `PK`/`SK`) e `service/pii.py` PiiMasker (NFR2.1/NFR2.2). *(Contract 5, NFR2.1, NFR2.2, NFR2.5)*
- [x] **Step 4** — Testes das camadas de dados/segurança (test-after). *(Testing Contract)*
- [x] **Step 5** — Lógica de negócio: `service/transcriber.py` WhisperTranscriber (ffmpeg→WAV 16k mono, faster-whisper PT-BR, import protegido, `model_factory` injetável), `service/telegram_gateway.py` (getFile/download/sendMessage com http injetado) e `service/router_gateway.py` (re-injeção via gateway injetado). *(FR1.2, NFR2.1)*
- [x] **Step 6** — Testes da lógica de negócio (test-after). *(Testing Contract)*
- [x] **Step 7** — Orquestrador + API: `service/voice_adapter.py` VoiceAdapter (valida Contract 3 → sessão → download → conversão → transcrição → PII → re-injeção; outcomes `ok`/`retry`/`drop`; fallback; logs estruturados) e `handler.py` SQS batch handler (`batchItemFailures`, DLQ-friendly, nunca derruba o lote). *(FR1.2, FR1.3, NFR4.1, NFR5.1, Contract 3)*
- [x] **Step 8** — Testes do orquestrador + integração (pipeline SQS end-to-end, lote parcial com drop/retry, wiring do `handler()`). *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (`requirements.txt`) e documentação/traceability (`source-manifest.json`, `traceability.json`). *(Contract 3, Contract 5)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | Contract 5, NFR2.1, NFR2.2, NFR2.5 |
| 4 | Testing Contract |
| 5 | FR1.2, NFR2.1 |
| 6 | Testing Contract |
| 7 | FR1.2, FR1.3, NFR4.1, NFR5.1, Contract 3 |
| 8 | Testing Contract |
| 9 | Contract 3, Contract 5 |
