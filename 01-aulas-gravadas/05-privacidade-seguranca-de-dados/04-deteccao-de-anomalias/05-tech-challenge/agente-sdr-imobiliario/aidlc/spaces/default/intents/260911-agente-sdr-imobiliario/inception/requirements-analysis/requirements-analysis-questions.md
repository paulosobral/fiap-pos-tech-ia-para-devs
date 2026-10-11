# Requirements Analysis Questions

## Sources

- [desc] Project description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [doc] PRD preliminar: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`
- [doc] Enunciado do desafio: `documentos/POSTECH - Hacka Agente_SDR_Imobiliario - Fase 5.md`
- [doc] Transcrição da mentoria: `documentos/transcricao-mentoria-21.md`

> Análise de requisitos baseada no PRD preliminar existente e nos documentos do desafio.

---

## Q1. O PRD preliminar captura adequadamente todos os requisitos do desafio?

O PRD preliminar (versão 1.10) é muito detalhado e cobre os aspectos técnicos, funcionais e não funcionais. Baseado nas suas respostas anteriores no AI-DLC, houve refinamentos importantes (públicos-alvo, prioridades, métricas, cronograma).

- A. Sim, o PRD está completo e as decisões do AI-DLC refinaram prioridades sem mudar o escopo fundamental
- B. Parcialmente, há ajustes importantes que precisam ser formalizados nos requisitos
- C. Não, há requisitos importantes faltando
- X. Other (especificar)

[Answer]: A

---

## Q2. A arquitetura proposta no PRD é adequada para a POC?

A arquitetura proposta é 100% serverless na AWS com Lambda, API Gateway, DynamoDB, S3, etc. Você confirmou na fase Feasibility: Python (LangGraph/LangChain, FastAPI, Streamlit), AWS serverless, LLM via OpenRouter (Claude 3.5 Haiku via LiteLLM).

- A. Sim, a arquitetura está adequada para a POC e alinhada com as decisões de stack
- B. Precisa de ajustes menores (especificar)
- C. Precisa de revisão significativa (especificar)
- X. Other (especificar)

[Answer]: A

---

## Q3. Os requisitos não funcionais (NFRs) estão bem definidos?

**Inconsistência encontrada no PRD**: NF-03 define "Primeira resposta < 4s", mas os KPIs de sucesso (linha 660) definem "Tempo de 1ª resposta < 10s". Você escolheu < 10s no AI-DLC, alinhado com os KPIs.

- A. Manter < 10s (alinhado com KPIs do PRD e sua escolha AI-DLC)
- B. Ajustar para < 4s (mais agressivo, alinhado com NF-03 do PRD)
- C. Definir outro valor (especificar)
- X. Other (especificar)

[Answer]: A

---

## Q4. Os cenários de usuário (C1-C7) estão bem definidos?

O PRD define 7 cenários: compra, locação corporativa, investimento, follow-up, anomalia, roleta de distribuição e contato via e-mail/portal. Você definiu prioridades: must-have (núcleo conversacional, anomalias, CRM HubSpot, voice), nice-to-have (roleta/esteira).

- A. Sim, os cenários estão completos e as prioridades estão claras
- B. Alguns cenários precisam de mais detalhes
- C. Cenários insuficientes ou faltando
- X. Other (especificar)

[Answer]: A

---

## Q5. A estratégia de segurança e privacidade (LGPD) está adequada?

O PRD define PII masking, criptografia KMS, consentimento, retenção TTL 90 dias, etc. Você confirmou na fase Feasibility: LGPD (PII mascarada, guardrails, trilha de auditoria) + nenhum requisito regulatório específico além do bom senso de segurança.

- A. Sim, a estratégia de segurança está adequada para a POC
- B. Precisa de reforços em áreas específicas
- C. Estratégia insuficiente para o contexto B2B
- X. Other (especificar)

[Answer]: A

---

## Q6. Há restrições técnicas ou de negócio que precisam ser esclarecidas?

Você definiu: prazo 12 de outubro (preferencialmente antes para gravação do vídeo), stack Python/AWS serverless/OpenRouter, conta AWS dedicada, custo baixo (~R$15/mês), integrações (HubSpot MCP demo + Private App REST, base imóveis simulada, Telegram).

- A. Não, as restrições estão claras e bem definidas
- B. Sim, há restrições que precisam de esclarecimento
- X. Other (especificar)

[Answer]: A

---

## Q7. O escopo está bem definido para o prazo do hackathon?

Você definiu prioridades claras: must-have (núcleo conversacional, anomalias, CRM HubSpot, voice), nice-to-have (roleta/esteira), sequenciamento dependency-first, e prazo ideal antes da gravação do vídeo.

- A. Sim, o escopo é realista com as prioridades definidas
- B. O escopo está um pouco amplo, pode precisar de priorização
- C. O escopo é muito ambicioso para o prazo
- X. Other (especificar)

[Answer]: A

---

## Assumptions & Open Questions

- O PRD preliminar é a fonte principal de requisitos
- A POC deve ser funcional e demonstrável, não produto final
- A prioridade é entregar todos os diferenciais do desafio (RAG, memória, multiagentes, etc.)
- A arquitetura serverless AWS é um requisito do desafio
- **Decisões AI-DLC que refinam o PRD**: priorização explícita (must-have vs nice-to-have), público-alvo expandido (investidor PF como parcela menor), métricas claras (resolvendo inconsistência interna PRD: 4s vs 10s)

## Review

Este arquivo é o registro das perguntas de análise de requisitos. As respostas alimentarão o documento de requisitos consolidado.

## Consolidated Summary Confirmation

Após verificação sistemática, suas decisões do AI-DLC refinaram o PRD: priorização explícita (must-have vs nice-to-have), público-alvo expandido (investidor PF como parcela menor), métricas consistentes (resolvendo inconsistência 4s vs 10s). Stack, integrações e segurança alinhados.

Does this all look correct before I generate the requirements artifact?

- Looks correct
- Request changes

[Answer]: Looks correct