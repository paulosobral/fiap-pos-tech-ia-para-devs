# Rough Mockups Questions

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [scope] Workflow-selected scope: `feature`.
- [assumption] Consome: `intent-statement` (intent-capture), `scope-document`, `intent-backlog` (scope-definition).

> Regras de operação confirmadas pelo usuário em conversa:
> - Fonte principal de entrada: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`.
> - Extrair o máximo que o PRD permitir; quando faltar informação necessária, retornar ao usuário para conferência antes de avançar de fase.

---

## Q1. Pontos de entrada e telas-chave

Quais as principais superfícies de UI da POC? (Selecione as que se aplicarem)

- A. Dashboard (Streamlit) — visão do corretor/SDR e do gestor (KPIs, leads, esteira Kanban, anomalias)
- B. Conversa no Telegram — interface do cliente final (atendimento conversacional)
- C. MCP Inspector — demo única do CRM HubSpot (não é UI do produto)
- D. Apenas dashboard — o resto é backend/API
- X. Other (especificar)

[Answer]: A, B e C

## Q2. Fluxo principal (happy path)

Qual o fluxo central a desenhar? (Selecione um)

- A. Cliente → Telegram → qualificação → rota na roleta → handoff ao corretor → corretor vê lead no dashboard
- B. Corretor/SDR → dashboard → filtra leads → vê timeline → age (agenda/atualiza status)
- C. Gestor → dashboard → vê KPIs e anomalias
- D. Ambos (fluxo do cliente + fluxo do corretor no dashboard)
- X. Other (especificar)

[Answer]: A, B (Corretor/SRD também importa leads via MCP HubSpot) e C

## Q3. Hierarquia de informação do dashboard

Qual a hierarquia de informação nas telas do dashboard? (Selecione o que se aplica)

- A. Topo: KPIs agregados (leads, qualificados, intenção, tempo de resposta)
- B. Meio: esteira Kanban (status dos leads) + timeline do lead
- C. Lateral/aba: alertas de anomalia
- D. Filtros por período/canal/status
- X. Other (especificar)

[Answer]: A, B, C e D

## Q4. Brand/design system

Há diretrizes de marca, design system ou padrões de UI a seguir? (Selecione um)

- A. Não — POC, usar padrão limpo e neutro (Streamlit default)
- B. Sim — identidade W Levitt (cores/logo) a aplicar no dashboard
- C. Sim — design system existente a seguir
- D. Não definido ainda
- X. Other (especificar)

[Answer]: A

## Q5. Form factor / dispositivos

Quais form factors/devices devem ser suportados? (Selecione o que se aplica)

- A. Desktop (dashboard Streamlit)
- B. Mobile (Telegram — responsivo por natureza)
- C. Apenas desktop — POC de demonstração
- X. Other (especificar)

[Answer]: A. OBS: Via telegram é o app existente, não precisa desenvolver nenhuma UI

## Q6. Acessibilidade

Há requisitos de acessibilidade conhecidos? (Selecione um)

- A. Nenhum específico — POC acadêmica; aplicar boas práticas básicas (headings, landmarks, contraste)
- B. WCAG AA — obrigatório
- C. Suporte a leitor de tela — obrigatório
- D. Navegação só por teclado — obrigatório
- X. Other (especificar)

[Answer]: A

---

## Assumptions & Open Questions

None.

## Review

Este arquivo é o registro de perguntas da etapa Rough Mockups & Concept Visualization. As respostas assinaladas guiarão os wireframes e o user-flow.

## Consolidated Summary Confirmation

- Looks correct
- Request changes

[Answer]: Looks correct