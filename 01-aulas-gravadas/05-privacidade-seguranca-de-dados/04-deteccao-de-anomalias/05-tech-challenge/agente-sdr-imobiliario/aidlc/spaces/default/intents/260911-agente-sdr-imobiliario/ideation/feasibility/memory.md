# Feasibility & Constraints — Memory (Diário)

> Estágio Feasibility & Constraints (Ideation), lead aidlc-architect-agent, suporte aws-platform + compliance.

## What was done

- Carregado run-stage de feasibility via `orchestrate continue` (bundle sha256:02b1b831...).
- Modo de interação escolhido pelo usuário: "I'll edit the file" (2).
- `feasibility-questions.md` (Q1–Q6) respondidas pelo usuário:
  - Q1: A, B, C (HubSpot via MCP demo + Private App REST; base imóveis simulada; Telegram POC / WhatsApp roadmap)
  - Q2: A, C (LGPD — PII mascarada, guardrails, trilha; nenhum outro requisito específico)
  - Q3: A, B, C, D (Python LangGraph/FastAPI/Streamlit; AWS serverless; OpenRouter Haiku via LiteLLM; time com capacidade de build)
  - Q4: A (prazo 12/10, ideal antes para gravação do vídeo) + D (sem restrição além do bom senso)
  - Q5: A (nenhum bloqueador organizacional)
  - Q6: A, B (conta AWS dedicada; Lambda/DynamoDB/S3/EventBridge/API Gateway/CloudWatch)
- Checkpoint `summary-confirmation` registrado (DECISION_RECORDED → `[Answer]: Looks correct` → SUMMARY_CONFIRMATION_RECORDED, auth `51a021fbfb80bec923c798d05dbe1d5719cb1d6d2ef9a976e6a3005f35f003d5`).
- Pareceres dos agentes (ensemble): aws-platform (viabilidade técnica/custo AWS) e compliance (LGPD). Veredito conjunto: VIÁVEL, sem bloqueio.
- Artefatos gerados: feasibility-assessment.md, constraint-register.md, raid-log.md.
- Sensores: required-sections (fire af9d068b, passed), upstream-coverage (fire 46d23150, passed, note "script-error: exit-1").
- Learnings: runtime compile + surface → nenhum candidato (memory_entries_total 0).

## Decisions

- POC 100% em dado sintético end-to-end (imóveis + leads + CRM) para reduzir superfície LGPD — recomendação compliance adotada.
- SSM Parameter Store no lugar de Secrets Manager (custo R$0 vs ~R$7–9/mês).
- Modelo LLM econômico (DeepSeek V3) na POC; Claude Haiku = upgrade documentado.
- EventBridge Scheduler (não Step Functions) para follow-up.
- Região us-east-1 na POC; sa-east-1 no roadmap.
- MoSCoW sequencial: núcleo primeiro, congelar infra ≥1 semana antes da gravação.
- Não alegar "conformidade LGPD" nem "trilha de auditoria LGPD completa" — alegar "segurança e privacidade por design (LGPD)".

## Blockers

- Nenhum bloqueador. Ressalvas: pin de model dos agentes continua cacheado na sessão (dispatch via `aidlc-product-lead-agent` falha; usar `general`); upstream-coverage passou com note "script-error: exit-1" (não bloqueante).

## Follow-ups

- Próximo estágio: Scope Definition (feasibility next_stage) — aguardar aprovação do gate e `orchestrate next`.
- Constraints CR-C1..C8, riscos R-01..R-09, issues I-01..I-04 e dependências D-01..D-05 devem alimentar requisitos/design nas fases seguintes.
- Verificar origem dos dados sintéticos (D-05) antes de construção.