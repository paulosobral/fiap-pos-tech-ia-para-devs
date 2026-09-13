# Build vs Buy — Agente SDR Imobiliário B2B (W Levitt)

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt …" (workflow-selected scope `feature`).
- [scope] Workflow-selected scope: `feature`.
- [Q6] [Q7] — respostas do usuário em `market-research-questions.md`.
- [assumption] PRD preliminar: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md` (§8.9 crm-adapter; §16 roadmap).

---

## 1. Postura geral

O usuário indicou [Q6] a postura de build-vs-buy para a POC:

- **A. Build: orquestração conversacional, qualificação, RAG, roleta, esteira Kanban, detecção de anomalias** (o núcleo do diferencial).
- **B. Buy/commodity: LLM via OpenRouter (Claude 3.5 Haiku via LiteLLM), infra AWS serverless**.

Ou seja: **build no que diferencia, buy no que é commodity**. O núcleo (orquestração, qualificação, RAG, roleta, Kanban, anomalias) é o diferencial competitivo e deve ser construído; o LLM e a infraestrutura serverless são commodities e devem ser comprados/alugados.

## 2. Avaliação por componente

| Componente | Build | Buy | Score (Build −2 a +2) | Decisão |
|---|---|---|---|---|
| Orquestração conversacional | Diferencial | LLM commodity | +2 | **Build** [Q6-A] |
| Qualificação + score explicável | Diferencial | Não há vendor maduro p/ B2B corporativo | +2 | **Build** [Q6-A] |
| RAG sobre catálogo de imóveis/clientes | Diferencial | FAISS/KB como lib | +2 | **Build** [Q6-A] |
| Roleta/rodízio de leads | Diferencial (regras por metragem) | Não há vendor específico | +2 | **Build** [Q6-A] |
| Esteira Kanban | Diferencial (integração com dashboard) | CRMs genéricos | +1 | **Build** [Q6-A] |
| Detecção de anomalias | Diferencial (plus de segurança) | Não há vendor específico | +2 | **Build** [Q6-A] |
| LLM (Claude 3.5 Haiku) | — | OpenRouter via LiteLLM | −2 | **Buy** [Q6-B] |
| Infra AWS serverless | — | Lambda/DynamoDB/S3/EventBridge | −2 | **Buy** [Q6-B] |
| CRM (HubSpot) | — | MCP (demo única) + Private App REST | −2 | **Buy/plugar** [Q6-C] |
| Observabilidade/CI | — | Ferramentas padrão | −1 | **Buy** [Q6-D] |

### Regra de decisão aplicada

> "Se a capacidade não é o seu diferencial central e existe um vendor maduro, default para Buy. Build apenas quando a capacidade é central para a vantagem competitiva ou quando nenhum vendor atende."

- **Build** nos componentes que são o diferencial (orquestração, qualificação, RAG, roleta, Kanban, anomalias) — não há vendor maduro especializado em imobiliário corporativo B2B.
- **Buy** no LLM (commodity, OpenRouter/Claude) e na infra AWS serverless (commodity).
- **Buy/plugar** no CRM (HubSpot): na POC, via MCP numa demonstração única ao vivo + Private App REST no `crm-adapter` de fundo [Q6-C][intent-statement A6].

## 3. CRM — detalhe do build-vs-buy

O CRM é o único componente com nuance de build-vs-buy [Q6-C][intent-statement A6]:

- **Não construir um CRM** — o HubSpot é o sistema de registro da W Levitt (e o `crm-adapter` já desacopla o fluxo, PRD §8.9).
- **Plugar via MCP**: HubSpot real numa demonstração única ao vivo (MCP auth app + MCP Inspector, OAuth/PKCE gerenciado pela ferramenta; redirect localhost) para o vídeo de apresentação.
- **Adapter de fundo (SQS → Lambda)**: simulado (default) + HubSpot real via Private App token (REST) — sem OAuth em Lambda assíncrona.
- **MCP-HubSpot em produção** (agente embutido / adapter MCP) fica no roadmap pós-POC.

## 4. TAM / SAM / SOM

O usuário indicou [Q7] o dimensionamento de mercado:

- **B. SAM: imobiliárias corporativas/comerciais B2B em São Paulo**.

O foco da POC é o **SAM** — imobiliárias corporativas/comerciais B2B em São Paulo — com a W Levitt como primeiro cliente (SOM). Não se dimensiona TAM global nesta etapa: a POC é demonstração (hackathon) e validação com um cliente real [Q7][Q8].

| Nível | Definição | Aplicação à POC |
|---|---|---|
| TAM | Mercado de atendimento/vendas imobiliárias com IA no Brasil | Contexto; não dimensionado nesta etapa [Q7] |
| **SAM** | Imobiliárias corporativas/comerciais B2B em São Paulo | **Foco da POC** [Q7-B] |
| SOM | W Levitt como primeiro cliente | Validação com 1 cliente real [Q7-C][Q8] |

## 5. Riscos e mitigações (build-vs-buy)

| Risco | Mitigação |
|---|---|
| LangGraph/frameworks evoluem | Feature flags; abstração fina da camada de canal (PRD §14) |
| Parada do OpenRouter | Fallback para Bedrock via LiteLLM (troca por config) (PRD §14) |
| Custo acima do budget | Limite de tokens, monitor semanal, alarme de limites (PRD §14) |
| OAuth em Lambda assíncrona (CRM) | Evitado: MCP só na demo única; adapter usa simulado + Private App REST |
| Dependência de vendor de LLM | Modelo agnóstico via LiteLLM (OpenRouter ↔ Bedrock) |

## 6. Conclusão

- **Build no núcleo do diferencial**: orquestração, qualificação proprietária + score explicável, RAG, roleta, esteira Kanban e detecção de anomalias [Q6-A].
- **Buy no commodity**: LLM (OpenRouter/Claude via LiteLLM) e infra AWS serverless [Q6-B].
- **Buy/plugar no CRM**: HubSpot via MCP (demo única) + Private App REST no adapter [Q6-C].
- **Foco de mercado**: SAM de imobiliárias corporativas/comerciais B2B em São Paulo, com a W Levitt como primeiro cliente [Q7-B][Q7-C][Q8].