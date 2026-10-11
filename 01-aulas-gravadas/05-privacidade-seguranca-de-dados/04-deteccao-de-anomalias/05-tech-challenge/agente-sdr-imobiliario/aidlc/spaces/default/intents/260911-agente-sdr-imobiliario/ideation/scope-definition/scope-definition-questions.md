# Scope Definition Questions

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [scope] Workflow-selected scope: `feature`.
- [assumption] Consome: `intent-statement` (intent-capture), `feasibility-assessment`, `constraint-register` (feasibility).

> Regras de operação confirmadas pelo usuário em conversa:
> - Fonte principal de entrada: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`.
> - Extrair o máximo que o PRD permitir; quando faltar informação necessária, retornar ao usuário para conferência antes de avançar de fase.

---

## Q1. Escopo mínimo viável (MVP)

Qual o escopo mínimo que entrega valor e cabe no prazo (12/10, ideal antes para gravação do vídeo)? (Selecione o que se aplica)

- A. Núcleo conversacional: canal Telegram → router → RAG (imóveis) → qualificação → handoff ao corretor + dashboard
- B. Roleta de leads + esteira Kanban (timeline no dashboard)
- C. Detecção de anomalias (IF+PCA+GLR+Autoencoder, job diário, alerta dashboard)
- D. CRM (HubSpot): MCP numa demo única + crm-adapter (SQS→Lambda, simulado default + Private App REST)
- E. Follow-up com contexto + calendário (convite ICS)
- F. Voice (faster-whisper) + ingestão de e-mail (SES, cenário C7)
- X. Other (especificar)

[Answer]: A, B, C, D e F (apenas o faster whisper para comunicação via voz no telegram)

## Q2. Must-have vs nice-to-have

Quais capacidades são obrigatórias (must-have) para a entrega e quais são desejáveis (nice-to-have)? (Selecione must-have)

- A. Must-have: núcleo conversacional (canal→router→RAG→qualificação→handoff→dashboard)
- B. Must-have: roleta + esteira Kanban + timeline
- C. Must-have: detecção de anomalias (diferencial da POC)
- D. Must-have: CRM HubSpot (MCP demo + adapter)
- E. Must-have: follow-up com contexto + calendário
- F. Nice-to-have: voice + ingestão de e-mail (C7)
- X. Other (especificar)

[Answer]: Must-have: A, C, D e F (apenas voice). Nice-to-have: B

## Q3. Dependências entre capacidades

Há dependências entre as capacidades que condicionem a ordem de construção? (Selecione o que se aplica)

- A. Núcleo conversacional é pré-requisito de tudo (roleta, anomalias, CRM, follow-up dependem dele)
- B. Roleta/esteira dependem do núcleo (rotas e status de leads)
- C. Anomalias dependem do núcleo (dados de sessão) e da esteira (restrição de agendamento)
- D. CRM e follow-up dependem do núcleo e da qualificação
- E. Voice/C7 são independentes e podem ser cortados sem afetar o núcleo
- X. Other (especificar)

[Answer]: A, B, C e D

## Q4. Sequenciamento

Qual a preferência de sequenciamento das capacidades? (Selecione um)

- A. Risk-first — atacar primeiro o que tem mais risco técnico (voice, anomalias, CRM)
- B. Value-first — entregar primeiro o que agrega mais valor ao cliente (núcleo conversacional)
- C. Dependency-first — seguir a ordem de dependências (núcleo → roleta/esteira → anomalias/CRM/follow-up)
- D. MoSCoW sequencial — núcleo primeiro (must-have), depois nice-to-have; congelar infra ≥1 semana antes da gravação
- X. Other (especificar)

[Answer]: C

## Q5. Prazos por capacidade

Há prazos rígidos ligados a capacidades específicas? (Selecione o que se aplica)

- A. Núcleo conversacional + dashboard: obrigatório até a data da gravação do vídeo
- B. Roleta + esteira + anomalias + CRM: idealmente antes da gravação (diferenciais do pitch)
- C. Follow-up + voice + C7: se couberem no prazo, senão ficam para depois
- D. Data da gravação é o limite rígido (não 12/10)
- E. Sem prazo rígido por capacidade — apenas o 12/10
- X. Other (especificar)

[Answer]: D

---

## Assumptions & Open Questions

None.

## Review

Este arquivo é o registro de perguntas da etapa Scope Definition & Prioritization. As respostas assinaladas guiarão o scope-document e intent-backlog.

## Consolidated Summary Confirmation

- Looks correct
- Request changes

[Answer]: Looks correct