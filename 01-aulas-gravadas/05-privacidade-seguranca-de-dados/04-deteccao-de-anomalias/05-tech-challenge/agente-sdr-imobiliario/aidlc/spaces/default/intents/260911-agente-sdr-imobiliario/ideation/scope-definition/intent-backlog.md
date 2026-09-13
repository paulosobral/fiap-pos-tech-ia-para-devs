# Intent Backlog — Agente SDR Imobiliário (W Levitt)

> Estágio Scope Definition & Prioritization (Ideation). Proto-Units priorizadas (MoSCoW). Sequenciamento: dependency-first (Q4-C). Limite rígido: data da gravação do vídeo (Q5-D).

## Must-have (núcleo, ordem de construção)

| # | Proto-Unit | Descrição | Depende de | Prioridade |
|---|---|---|---|---|
| SC-01 | Núcleo conversacional | Canal Telegram → conversation router → RAG (imóveis) → qualificação → handoff ao corretor + dashboard | — | P0 |
| SC-02 | Detecção de anomalias | IF + PCA + GLR + Autoencoder; job diário (Lambda/EventBridge); alerta dashboard; restrição de agendamento p/ leads suspeitos | SC-01 | P1 |
| SC-03 | CRM HubSpot | MCP numa demo única (Inspector, OAuth/PKCE) + crm-adapter (SQS→Lambda, simulado default + Private App REST) | SC-01 | P1 |
| SC-04 | Voice (faster-whisper) | Transcrição de voz no Telegram via faster-whisper em Lambda layer; desacoplado via SQS | SC-01 | P1 |

## Nice-to-have

| # | Proto-Unit | Descrição | Depende de | Prioridade |
|---|---|---|---|---|
| SC-05 | Roleta + esteira Kanban | Rotas configuráveis; timeline do lead no dashboard (entrada, qualificação, rota, status, anomalias) lida do DynamoDB | SC-01 | P2 |

## Fora do escopo da POC (roadmap pós-POC)

| # | Proto-Unit | Motivo |
|---|---|---|
| SC-06 | Follow-up com contexto + calendário (ICS) | Excluído por decisão do usuário (dor central, mas fora da POC) |
| SC-07 | Ingestão de e-mail (SES, cenário C7) | Excluído; apenas voice no item voice |
| SC-08 | WhatsApp | Roadmap |
| SC-09 | Reporte formal periódico | Dashboard em tempo real cobre agregado |
| SC-10 | Auditoria formal imutável | POC tem registro/visibilidade (timeline DynamoDB) |
| SC-11 | Bedrock produção / sa-east-1 | Roadmap |
| SC-12 | MCP-HubSpot embutido em produção | Roadmap |

## Notas de priorização

- **P0**: núcleo conversacional — pré-requisito de tudo (Q3-A) e obrigatório até a gravação (Q5-A).
- **P1**: anomalias, CRM, voice — must-have (Q2), dependem do núcleo (Q3-C/D), idealmente antes da gravação.
- **P2**: roleta/esteira — nice-to-have (Q2), mas incluída no escopo mínimo (Q1-B).
- **Congelar infra ≥1 semana antes da gravação** (recomendação feasibility).
- MoSCoW sequencial: núcleo primeiro, depois os demais; cortar o que não couber sem afetar o núcleo.