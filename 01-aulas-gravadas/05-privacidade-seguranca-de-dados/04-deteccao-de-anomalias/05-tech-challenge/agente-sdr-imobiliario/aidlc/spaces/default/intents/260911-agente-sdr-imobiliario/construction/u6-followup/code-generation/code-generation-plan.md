# Code Generation Plan — u6-followup

> Unidade U6 (Followup — Follow-up Automático). Escopo: `feature` · Estratégia: `standard` · Metodologia: `test-after` (Testing Contract do time). Greenfield.
> Fontes: contract-summary.md (Contratos 1/5), unit-of-work.md (U6), requirements.md (FR8.1–FR8.3, NFR5.1, NFR2.1, NFR4.1), referência de convenções: u5-anomaly (leitor do Contrato 5/6 e job EventBridge), u4-async-ingest (SessionWriter, guarda de duplicidade, wiring por DI), u2-async-voice (TelegramGateway com http injetado), u1-core-conversation (schema de sessões/conversas).

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

- [x] **Step 1** — Estrutura do projeto e configuração de produção (layout `apps/followup/{handler.py, service/, infra/, tests/, requirements.txt}` conforme PRD §7.4). *(História: setup da unidade U6)*
- [x] **Step 2** — Bootstrap do runner de testes (pytest + pytest-cov já no `.venv` da raiz; path bootstrap via `tests/conftest.py`, pyproject da raiz intocado) e registro do comando exato com escopo da unidade. *(Testing Contract: runner antes do primeiro teste)*
- [x] **Step 3** — Camada de dados: `infra/conversation_store.py` ConversationStore (leitor do Contrato 5 — scan `begins_with(SK, CONV#)` para conversas e `begins_with(SK, PROFILE)` para o mapa `lead_id → telegram_user_id/intent`, unmarshal de tipos e campos JSON, itens sem identificadores ignorados com log) e `infra/followup_state.py` FollowupStateStore (estado próprio de cadência `SK=FOLLOWUP`: passo enviado, `last_followup_at`, próximo passo da cadência), `infra/silence_window.py` SilenceWindow (janela configurável por env, clock injetável, suporte a janela que cruza meia-noite). *(FR8.2, FR8.3, Contrato 5)*
- [x] **Step 4** — Testes das camadas de dados (test-after). *(Testing Contract)*
- [x] **Step 5** — Cadência e mensagem: `service/cadence.py` CadenceCalculator (dias 2/5/9 configuráveis via env, clock injetável, retomada a partir do último passo enviado, expiração após o último dia; simula os wait states do Step Functions na POC), `service/message_builder.py` FollowupMessageBuilder (mensagem PT-BR por passo da cadência com contexto resumido — passo, intenção, estado — sem jamais receber/enviar PII), `service/telegram_gateway.py` TelegramGateway (sendMessage do Contrato 1 via http injetado, erro vira `GatewayError` para retry no próximo tick). *(FR8.1, FR8.2, Contrato 1)*
- [x] **Step 6** — Testes de cadência, mensagem e gateway (test-after). *(Testing Contract)*
- [x] **Step 7** — Guarda + orquestrador + handler: `service/duplicate_guard.py` DuplicateGuard (supressão por passo já enviado — idempotência do tick — e por resposta do lead após o último follow-up), `service/followup.py` FollowupService (pipeline: conversa mais recente por lead → passo vencido → guarda → janela de silêncio → contexto → mensagem → envio → persistir próximo passo; drop explícito com log para lead inválido/sem contexto/sem canal; clock injetável), `handler.py` (wiring EventBridge com boto3/env/requests, DI completo, `FOLLOWUP_CADENCE_DAYS`/`SILENCE_WINDOW_START`/`SILENCE_WINDOW_END` configuráveis). *(FR8.1, FR8.2, FR8.3, NFR5.1, NFR2.1, NFR4.1)*
- [x] **Step 8** — Testes do orquestrador + integração (pipeline ponta a ponta com stores fake, janela fechada difere sem envio, guarda de resposta, idempotência do passo, drop sem contexto, PII-safe nos logs, wiring do `handler()` com EventBridge, env ausente). *(Testing Contract)*
- [x] **Step 9** — Configuração de build/deploy (`requirements.txt`) e documentação/traceability (`source-manifest.json`, `traceability.json`). *(Contrato 1, FR8)*

## Rastreabilidade passo → requisito

| Step | Requisito(s) |
|------|--------------|
| 1 | — (estrutura) |
| 2 | Testing Contract |
| 3 | FR8.2, FR8.3, Contrato 5 |
| 4 | Testing Contract |
| 5 | FR8.1, FR8.2, Contrato 1 |
| 6 | Testing Contract |
| 7 | FR8.1, FR8.2, FR8.3, NFR5.1, NFR2.1, NFR4.1 |
| 8 | Testing Contract |
| 9 | Contrato 1, FR8 |
