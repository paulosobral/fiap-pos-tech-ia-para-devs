# Market Research Questions

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [scope] Workflow-selected scope: `feature`.
- [assumption] Consome: `intent-statement` da etapa Intent Capture.

> Regras de operação confirmadas pelo usuário em conversa:
> - Fonte principal de entrada: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`.
> - Extrair o máximo que o PRD permitir; quando faltar informação necessária, retornar ao usuário para conferência antes de avançar de fase.

---

## Q1. Concorrentes diretos

Quais produtos/soluções concorrentes diretos existem no mercado para o Agente SDR Imobiliário B2B? (Selecione os que se aplicarem)

- A. Lais.ai — SDR de IA para imobiliário residencial (WhatsApp)
- B. Maya (PLAZA) — atendimento omnichannel para imobiliárias residenciais
- C. Squad — solução de atendimento/vendas com IA para imobiliárias
- D. Chatbots genéricos de atendimento (ex.: Zendesk, Intercom, Blip) configurados para imobiliárias
- E. Assistência humana tradicional (corretores/SDRs respondendo manualmente) como "concorrente" de status quo
- X. Other (especificar)

[Answer]: O cliente recomendou as opções A e B (focados mais no público de imobiliárias). Ele comentou sobre a opção C para conhecimento (mais genérico, atende diversos segmentos).

## Q2. Concorrentes indiretos / substitutos

Que soluções indiretas ou substitutas resolvem o mesmo problema de outra forma? (Selecione os que se aplicarem)

- A. CRMs com automação de follow-up (ex.: HubSpot, Pipedrive, RD Station) usados pela própria W Levitt
- B. Portais imobiliários com chat próprio (Zap, VivaReal, OLX) que capturam o lead antes da imobiliária
- C. Planilhas + trabalho manual de SDR/corretor (status quo)
- D. Serviços de telemarketing/tribo de qualificação terceirizada
- E. Não há substituto relevante — o problema é específico de imobiliário corporativo B2B
- X. Other (especificar)

[Answer]: C

## Q3. Posicionamento competitivo

Qual deve ser o diferencial competitivo central do Levitt.AI na POC? (Selecione os que se aplicarem)

- A. Especialização em imóveis corporativos/comerciais B2B (lajes, andares, salas) — nicho não atendido por Lais/Maya/Squad
- B. Fully serverless e baixo custo (~R$ 15/mês em POC via OpenRouter)
- C. Detecção de anomalias (IF + PCA + GLR + Autoencoder) como plus de segurança/diferencial
- D. Qualificação proprietária + RAG + score explicável (não só fluxo fixo)
- E. Atendimento 24×7 com IA + humano no mesmo número
- X. Other (especificar)

[Answer]: B e D

## Q4. Tendências de mercado

Quais tendências de mercado ou mudanças regulatórias são relevantes para a POC? (Selecione os que se aplicarem)

- A. Adoção crescente de IA generativa em atendimento/vendas imobiliárias (2024–2026)
- B. LGPD e proteção de dados (PII mascarada, guardrails, trilha de auditoria) como requisito
- C. Migração de canais para WhatsApp Business API (Meta) e omnichannel
- D. Expectativa de resposta imediata (tempo de 1ª resposta) como table-stake
- E. Nenhuma tendência relevante — contexto estável
- X. Other (especificar)

[Answer]: A

## Q5. Table-stakes vs diferentes

O que os clientes esperam como mínimo (table-stake) vs. o que seria diferencial? (Selecione os que se aplicarem)

- A. Table-stake: atendimento conversacional com resposta imediata e follow-up automático
- B. Table-stake: integração com CRM e handoff qualificado ao corretor
- C. Diferencial: especialização B2B corporativa (entendimento de lajes/andares/salas)
- D. Diferencial: score explicável + detecção de anomalias
- E. Table-stake e diferenciais ainda não definidos — decidir nesta etapa
- X. Other (especificar)

[Answer]: A

## Q6. Build vs Buy

Para a POC, qual a postura de build-vs-buy para os componentes? (Selecione os que se aplicarem)

- A. Build: orquestração conversacional, qualificação, RAG, roleta, esteira Kanban, detecção de anomalias (núcleo do diferencial)
- B. Buy/commodity: LLM via OpenRouter (Claude 3.5 Haiku via LiteLLM), infra AWS serverless
- C. Buy/plugar: CRM via MCP (HubSpot) — demo única + Private App REST no adapter
- D. Buy: ferramentas de observabilidade/CI padrão (não reinventar)
- E. Tudo build — sem dependência de terceiros além do LLM
- X. Other (especificar)

[Answer]: A e B

## Q7. TAM/SAM/SOM

Qual o tamanho de mercado a considerar para a POC? (Selecione o que se aplica)

- A. TAM: mercado de atendimento/vendas imobiliárias com IA no Brasil
- B. SAM: imobiliárias corporativas/comerciais B2B em São Paulo
- C. SOM: W Levitt (uma consultoria) como primeiro cliente — POC é para validar com 1 cliente
- D. Não dimensionar agora — POC é demonstração acadêmica (hackathon)
- E. Não definido ainda — decidir nesta etapa
- X. Other (especificar)

[Answer]: B

## Q8. Adoção de mercado

Qual o objetivo de mercado da POC em relação a adoção? (Selecione o que se aplica)

- A. Validar a solução com a W Levitt como cliente real (prova de conceito)
- B. Posicionar como diferencial competitivo do hackathon (demonstração/avaliação)
- C. Preparar para go-to-market pós-POC (roadmap: WhatsApp, omnichannel, multi-tenant)
- D. Não há objetivo de mercado — apenas entregar a POC acadêmica
- E. Não definido ainda
- X. Other (especificar)

[Answer]: B

---

## Assumptions & Open Questions

None.

## Review

Este arquivo é o registro de perguntas da etapa Market Research. As respostas assinaladas guiarão a geração do competitive-analysis, market-trends, build-vs-buy e da estratégia de diferenciação.

## Consolidated Summary Confirmation

- Looks correct
- Request changes

[Answer]: Looks correct