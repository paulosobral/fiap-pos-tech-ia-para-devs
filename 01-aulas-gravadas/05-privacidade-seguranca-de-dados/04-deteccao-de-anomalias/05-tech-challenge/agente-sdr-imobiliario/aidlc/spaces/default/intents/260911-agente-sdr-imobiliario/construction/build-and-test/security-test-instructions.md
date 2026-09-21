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
| PII mascarada antes do envio ao LLM (NFR3.1) | `apps/conversation-router/service/security_layer.py` + `test_pii_masker.py` |
| Consentimento na 1ª mensagem + registrado (NFR2.3, NFR3.2) | `test_sales_flow.py`, `test_entities.py`, `test_conversation_router.py` |
| Retenção TTL (NFR3.3) | `test_session_store.py` (u1/u4), `test_conversation_store.py` (u5) |
| Guardrails / denied topics / prompt injection / evasão PII (NFR7.1–7.3) | `test_guardrails.py` |
| PII-safe nos logs (NFR2.5) | `log_event` JSON sem PII — provado por `caplog` nas suítes |
| PII mascarada na saída CRM (NFR2.2-ish boundary u3) | `apps/crm-adapter/service/pii_mask.py` + `test_pii_mask.py` |

## Análise estática leve (executável)
```bash
# Nenhuma execução dinâmica de código / shell embutido:
grep -rnE "eval\(|exec\(|os\.system|subprocess\.[A-Za-z]*\(" apps --include='*.py' | grep -v tests/ || echo "CLEAN"
```

## Testes de segurança NÃO executáveis localmente (deferidos)
- **DAST/pen-test** contra API Gateway/Lambda real → **deployment-pipeline**.
- **Testes de consentimento em produção (Canary/trilha)** → **observability-setup**.
- **Demo MCP HubSpot com MCP Inspector (FR11.1)** — autenticação app + inspector: requer ambiente com credenciais → **deployment-execution** (demo única ao vivo).

## Registra deferral
- `NFR2.4` (trilha de auditoria dedicada e append-only) → ver matrix; mínimo local = logs JSON + session store auditável.
- `FR11.1` → **deployment-execution**.
