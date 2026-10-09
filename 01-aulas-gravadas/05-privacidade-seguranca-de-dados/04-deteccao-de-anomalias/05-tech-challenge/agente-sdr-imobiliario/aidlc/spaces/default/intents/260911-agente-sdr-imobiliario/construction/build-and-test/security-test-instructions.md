# Security Test Instructions — Agente SDR Imobiliário (POC)

Contexto de NFR: **NFR2.1–2.5** (LGPD: consentimento, logs JSON, PII-safe, trilha de auditoria), **NFR3.1–3.3** (PII mascarada antes do LLM, consent registrado, TTL de retenção), **NFR7.1–7.3** (guardrails/denied topics, prompt injection, evasão de PII).

## Suíte de segurança executável localmente (jest→pytest)
Todos os testes abaixo **já existem** e rodam dentro das suítes por unit; comando dedicado:
```bash
# PII masking/unmask + guardrails + consent (u1)
.venv/bin/python -m pytest \
  apps/conversation-router/tests/unit/test_pii_masker.py \
  apps/conversation-router/tests/unit/test_guardrails.py \
  apps/conversation-router/tests/unit/test_session_store.py \
  apps/conversation-router/tests/integration/test_conversation_router.py -q

# TTL / retenção (u1, u4, u5)
.venv/bin/python -m pytest \
  apps/conversation-router/tests/unit/test_session_store.py \
  apps/contact-ingest/tests/unit/test_session_store.py \
  apps/anomaly-detector/tests/unit/test_conversation_store.py -q

# PII masking na saída CRM (u3)
.venv/bin/python -m pytest apps/crm-adapter/tests/unit/test_pii_mask.py -q
```

## Cenários cobertos (evidência)
| Cenário | Onde é provado |
|---|---|
| PII (e-mail, telefone — digitado e falado —, CNPJ) mascarada antes do envio ao LLM (NFR3.1). **Nome não é mascarado** por decisão de produto (vem do perfil do Telegram) | `security_layer.py` + `test_pii_masker.py`; saída barrada por `check_output_leak` |
| Consentimento **explícito** (só "sim" claro grava; qualquer outra resposta reapresenta o pedido; "não" encerra) e registrado (NFR2.3, NFR3.2; ADR-017) | `test_conversation_polish.py`, `test_sales_flow.py`, `test_conversation_router.py` |
| Contato anotado só em código, depois do check de vazamento | `test_conversation_polish.py::TestContactConfirmation`, integração |
| Dados de lead no dashboard só com JWT Cognito (401 sem token); PII decifrada por KMS na leitura | `apps/dashboard-api/tests/unit/test_leads.py` |
| Sessão do dashboard: cookie com id opaco, refresh token só no servidor, TTL 12 h | `test_app_smoke.py::TestSessionVault` |
| Segredos HubSpot só em `secrets.local.env` (gitignored) → Secrets Manager; refresh token rotacionado na secret | `test_hubspot_mcp.py`, `.gitignore` |
| Retenção TTL (NFR3.3) | `test_session_store.py` (u1/u4), `test_conversation_store.py` (u5) |
| Guardrails / denied topics / prompt injection / evasão PII (NFR7.1–7.3) | `test_guardrails.py` |
| PII-safe nos logs (NFR2.5) | `log_event` JSON sem PII — provado por `caplog` nas suítes |
| PII mascarada na saída CRM (NFR2.2-ish boundary u3) | `apps/crm-adapter/service/pii_mask.py` + `test_pii_mask.py` |

## Análise estática leve (executável)
```bash
# Nenhuma execução dinâmica de código / shell embutido:
grep -rnE "eval\(|exec\(|os\.system|subprocess\.[A-Za-z]*\(" apps --include='*.py' | grep -v tests/
# Esperado: somente apps/voice-adapter/service/transcriber.py (ffmpeg com lista de argumentos, sem shell=True, com timeout).
```

## Testes de segurança NÃO executáveis localmente (deferidos)
- **DAST/pen-test** contra API Gateway/Lambda real → **deployment-pipeline**.
- **Testes de consentimento em produção (Canary/trilha)** → **observability-setup**.
- **Autorização do HubSpot MCP (FR11.1)**: validada manualmente em 2026-10-08 com `scripts/hubspot_authorize.py` (OAuth 2.1 + PKCE). Sem teste automatizado possível (exige navegador e conta).
- **DAST/authz do `/api/leads`** (acesso sem token/outro usuário contra a API real) → **deployment-execution**.

## Registra deferral
- `NFR2.4` (trilha de auditoria dedicada e append-only) → ver matrix; mínimo local = logs JSON + session store auditável.
- `FR11.1` → validado manualmente (ver acima); sem automação.
