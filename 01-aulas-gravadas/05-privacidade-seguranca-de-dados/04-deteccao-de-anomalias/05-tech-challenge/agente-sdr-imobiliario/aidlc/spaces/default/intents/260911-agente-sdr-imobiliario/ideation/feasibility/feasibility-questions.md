# Feasibility Questions

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [scope] Workflow-selected scope: `feature`.
- [assumption] Consome: `intent-statement` (intent-capture), `competitive-analysis`, `market-trends`, `build-vs-buy` (market-research).

> Regras de operação confirmadas pelo usuário em conversa:
> - Fonte principal de entrada: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`.
> - Extrair o máximo que o PRD permitir; quando faltar informação necessária, retornar ao usuário para conferência antes de avançar de fase.

---

## Q1. Integrações com sistemas existentes

Com quais sistemas existentes a POC deve integrar (ou simular)? (Selecione os que se aplicarem)

- A. CRM (HubSpot) — via MCP numa demo única + Private App REST no adapter
- B. Base de imóveis — simulada (JSON + S3), sem portal real
- C. Canal de atendimento — Telegram (POC); WhatsApp no roadmap
- D. Calendário — simulado (geração de convite ICS)
- E. Nenhuma integração real — tudo simulado na POC
- X. Other (especificar)

[Answer]: A, B e C

## Q2. Requisitos regulatórios/compliance

Há requisitos regulatórios ou de compliance para a POC? (Selecione os que se aplicarem)

- A. LGPD — PII mascarada, guardrails, trilha de auditoria (dados de leads)
- B. Residência de dados — dados dentro do Brasil
- C. Nenhum requisito regulatório específico além do bom senso de segurança
- D. Requisitos de segurança internos da W Levitt (a definir)
- E. Não se aplica — POC acadêmica com dados sintéticos
- X. Other (especificar)

[Answer]: A e C

## Q3. Stack e perfil do time

Qual a stack e o perfil de skill do time para a POC? (Selecione o que se aplica)

- A. Python (LangGraph/LangChain, FastAPI, Streamlit) — alinhado ao PRD preliminar
- B. AWS serverless (Lambda, DynamoDB, S3, EventBridge, API Gateway) — alinhado ao PRD
- C. LLM via OpenRouter (Claude 3.5 Haiku via LiteLLM)
- D. Time com experiência em Python e AWS (capacidade de build)
- E. Stack ainda não definida — decidir nesta etapa
- X. Other (especificar)

[Answer]: A, B, C e D

## Q4. Orçamento e prazo

Quais as restrições de orçamento e prazo? (Selecione o que se aplica)

- A. Prazo: entrega em 12 de outubro (hackathon FIAP Fase 5)
- B. Orçamento: baixo custo (~R$ 15/mês em POC via OpenRouter)
- C. Orçamento: limite de tokens e monitoramento de custo
- D. Sem restrição de orçamento além do bom senso
- E. Não definido ainda
- X. Other (especificar)

[Answer]: A (porém é bom terminar antes para a gravação do vídeo de apresentação gravado por mim) e D.

## Q5. Bloqueadores organizacionais

Há bloqueadores organizacionais (freeze de mudança, prioridades concorrentes)? (Selecione o que se aplica)

- A. Nenhum — projeto novo, sem sistemas legados críticos em produção
- B. Dependência de aprovação da W Levitt para acesso a dados reais
- C. Prioridades concorrentes do time
- D. Freeze de mudança em período do hackathon
- E. Não identificado ainda
- X. Other (especificar)

[Answer]: A

## Q6. Serviços/conta AWS

Quais serviços e contas AWS estão em uso ou planejados? (Selecione os que se aplicarem)

- A. Conta AWS dedicada/novo para a POC
- B. Lambda, DynamoDB, S3, EventBridge, API Gateway, CloudWatch
- C. Bedrock (produção/roadmap) vs OpenRouter (POC)
- D. Nenhuma conta AWS ainda — provisionar nesta etapa
- E. Não definido ainda
- X. Other (especificar)

[Answer]: A e B

---

## Assumptions & Open Questions

None.

## Review

Este arquivo é o registro de perguntas da etapa Feasibility & Constraints. As respostas assinaladas guiarão o feasibility-assessment, constraint-register e raid-log.

## Consolidated Summary Confirmation

- Looks correct
- Request changes

[Answer]: Looks correct