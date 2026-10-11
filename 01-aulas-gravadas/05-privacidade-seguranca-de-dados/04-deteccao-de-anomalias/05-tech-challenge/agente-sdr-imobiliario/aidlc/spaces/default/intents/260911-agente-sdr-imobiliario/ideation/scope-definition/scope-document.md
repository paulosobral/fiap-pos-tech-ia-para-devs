# Scope Document — Agente SDR Imobiliário (W Levitt)

> Estágio Scope Definition & Prioritization (Ideation). Fonte: intent-statement, feasibility-assessment, constraint-register, scope-definition-questions (Q1–Q5).

## In-Scope (POC)

### Núcleo conversacional (must-have)
- Canal Telegram (única via; WhatsApp fora do escopo)
- Conversation router → RAG (base de imóveis simulada, JSON+S3) → qualificação → handoff ao corretor
- Dashboard (Streamlit Community Cloud) com KPIs agregados

### Detecção de anomalias (must-have, diferencial da POC)
- IF + PCA + GLR + Autoencoder (técnicas das aulas)
- Job diário (Lambda, EventBridge Scheduler)
- Alerta no dashboard; restrição de agendamento para leads suspeitos

### CRM HubSpot (must-have)
- MCP numa demo única ao vivo (vídeo) via MCP Inspector (OAuth/PKCE, redirect localhost)
- crm-adapter de fundo: SQS → Lambda, simulado default + HubSpot real via Private App token (REST)

### Voice (must-have — apenas faster-whisper, comunicação por voz no Telegram)
- faster-whisper em Lambda layer; transcrição desacoplada via SQS
- Ingestão de e-mail (SES, cenário C7) **fora** — ver Out-of-Scope

### Roleta de leads + esteira Kanban (nice-to-have)
- Rotas configuráveis, timeline do lead no dashboard (entrada, qualificação, rota, status, anomalias) lida do DynamoDB

## Out-of-Scope (POC)

- **Follow-up com contexto + calendário (convite ICS)** — dor central (leads parados), mas **excluído da POC** por decisão do usuário; fica para roadmap pós-POC
- **Ingestão de e-mail (SES, cenário C7)** — excluída; apenas voice no item F
- WhatsApp (roadmap)
- Reporte formal periódico (semanal/mensal) — dashboard em tempo real cobre o agregado (roadmap)
- Auditoria formal imutável — POC tem "registro e visibilidade de operações" (timeline DynamoDB)
- Bedrock em produção / sa-east-1 (roadmap)
- MCP-HubSpot embutido em produção (roadmap)

## Boundary decisions

- **Dados sintéticos end-to-end** (imóveis + leads + CRM) para reduzir superfície LGPD (recomendação compliance adotada)
- **SSM Parameter Store** no lugar de Secrets Manager (custo R$0)
- **Modelo LLM econômico** (DeepSeek V3) na POC; Claude Haiku = upgrade documentado
- **EventBridge Scheduler** (não Step Functions) para follow-up futuro
- **Região us-east-1** na POC; sa-east-1 no roadmap
- **MoSCoW sequencial**: núcleo primeiro, congelar infra ≥1 semana antes da gravação
- **Limite rígido = data da gravação do vídeo** (não 12/10)

## Sequencing (Q4: dependency-first)

1. Núcleo conversacional (canal→router→RAG→qualificação→handoff→dashboard)
2. Roleta + esteira Kanban (nice-to-have)
3. Detecção de anomalias (depende do núcleo + esteira)
4. CRM HubSpot (MCP demo + adapter)
5. Voice (faster-whisper)

## Success metrics (do intent-statement)

- Tempo de primeira resposta < 10 s
- Taxa de leads qualificados ≥ 60% das conversas
- Intenção de compra/locação/investimento detectada ≥ 85%

## Constraints aplicáveis

- Técnicas: CT-01..11 (limite pacote Lambda, cold start, OpenRouter SPOF, custo, SSM, Scheduler, SES sandbox, Streamlit, MCP, região, retenção logs)
- Organizacionais: CO-01..07 (prazo, orçamento, sem bloqueadores, stack, base simulada, Telegram, conta AWS)
- Regulatórias: CR-C1..C8 (masking PII, consentimento, aviso privacidade, retenção/TTL, logs masking, dados sintéticos, segredos, teardown)