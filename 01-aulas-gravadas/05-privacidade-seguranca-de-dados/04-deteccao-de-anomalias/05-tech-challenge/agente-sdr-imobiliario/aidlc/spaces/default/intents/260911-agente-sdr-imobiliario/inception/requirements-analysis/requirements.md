# Requirements — Agente SDR Imobiliário B2B

## Análise de Intent

O objetivo é desenvolver uma **Prova de Conceito (POC)** funcional do Agente SDR Imobiliário B2B para a W Levitt, especializado em espaços corporativos (lajes, andares, salas comerciais) em São Paulo. A POC deve demonstrar atendimento conversacional humanizado, qualificação de leads, identificação de intenção (compra/locação/investimento), RAG sobre base de imóveis, follow-up automático com memória, agendamento de reuniões, handoff para corretores, dashboard com KPIs e detecção de anomalias.

**Problemas de negócio resolvidos:**
- Atendimento de primeira resposta não limitado ao horário comercial (chatbot 24×7)
- Follow-up de leads parados automatizado (sem depender de corretores humanos)
- Qualificação e triagem de leads sem consumir tempo do corretor especialista
- Identificação clara de intenção (compra/locação/investimento) com escala no primeiro atendimento

**Métricas de sucesso da POC:**
- Tempo de 1ª resposta < 10s
- Taxa de leads qualificados >= 60% das conversas
- Intenção detectada corretamente >= 85%

## Requisitos Funcionais

### FR1 — Atendimento Conversacional
O agente deve receber e processar mensagens de leads via Telegram com conversa natural e humanizada, não limitada a horário comercial.

**Sub-requisitos:**
- FR1.1: Receber mensagens de texto via webhook do Telegram
- FR1.2: Processar mensagens de voz (voice messages) com transcrição STT (faster-whisper PT-BR)
- FR1.3: Manter contexto conversacional contínuo entre mensagens
- FR1.4: Oferecer botões inline como atalhos, mas aceitar texto livre

### FR2 — Identificação de Intenção
O agente deve classificar a intenção do lead em: compra, locação corporativa, ou investimento.

**Sub-requisitos:**
- FR2.1: Detectar intenção no início do fluxo conversacional
- FR2.2: Coletar metragem, região, orçamento, prazo, nº de pessoas, decisor
- FR2.3: Diferenciar perfil B2B (principal) vs investidor PF (parcela menor)

### FR3 — Qualificação de Leads
O agente deve qualificar leads através de questionário adaptativo e scoring.

**Sub-requisitos:**
- FR3.1: Aplicar questionário adaptativo baseado no perfil detectado
- FR3.2: Calcular score de prontidão e urgência
- FR3.3: Identificar ticket médio e expectativa de retorno (para investidores)

### FR4 — RAG sobre Base de Imóveis
O agente deve contextualizar com RAG sobre base simulada de imóveis corporativos sintéticos.

**Sub-requisitos:**
- FR4.1: Carregar índice FAISS em memória da base de imóveis (100–200 ofertas)
- FR4.2: Buscar top-k imóveis compatíveis com filtros do lead
- FR4.3: Nunca inventar imóveis que não estão na base (constraint via RAG)

### FR5 — Agendamento de Reuniões
O agente deve agendar reuniões/visitas com corretores especialistas.

**Sub-requisitos:**
- FR5.1: Validar data/hora disponível
- FR5.2: Gravar compromisso no calendário simulado
- FR5.3: Emitir convite ICS para o lead
- FR5.4: Notificar corretor via canal interno

### FR6 — Resumo Inteligente (Handoff)
O agente deve gerar resumo inteligente para o corretor quando o lead estiver qualificado.

**Sub-requisitos:**
- FR6.1: Gerar handoff em Markdown com gap, score, intenção, urgência, próximos passos
- FR6.2: Enviar mensagem interna + arquivo de resumo ao corretor
- FR6.3: Desmascarar PII apenas no handoff interno (uso criptografado)

### FR7 — Dashboard Mínimo
O agente deve entregar dashboard mínimo (1 página Streamlit) com KPIs e anomalias.

**Sub-requisitos:**
- FR7.1: Consumir `GET /api/kpis` com métricas de negócio
- FR7.2: Exibir linha de métricas: leads hoje/semana, tempo 1ª resposta p90, taxa qualificação, agendamentos
- FR7.3: Exibir gráficos: volume de intenções, leads distribuídos pela roleta
- FR7.4: Exibir tabela de anomalias com alertas 24h

### FR8 — Follow-up Automático
O agente deve fazer follow-up automático com memória conversacional.

**Sub-requisitos:**
- FR8.1: Implementar cadências configuráveis via EventBridge (dia 2, 5, 9)
- FR8.2: Manter contexto da última conversa ao retomar contato
- FR8.3: Respeitar janela de silêncio para evitar spam

### FR9 — Detecção de Anomalias
O agente deve detectar anomalias em conversas (módulo da fase).

**Sub-requisitos:**
- FR9.1: Job diário extrai features por conversa (volume, comprimento, sentimento, horários atípicos)
- FR9.2: Aplicar Isolation Forest + PCA + Autoencoder
- FR9.3: Emitir alerta no dashboard
- FR9.4: Restringir agendamento para leads suspeitos

### FR10 — Roleta de Distribuição
O agente deve distribuir leads qualificados para corretores via regras configuráveis.

**Sub-requisitos:**
- FR10.1: Aplicar regra "até 500 m² → rodízio dos consultores"
- FR10.2: Aplicar regra "acima de 500 m² → diretor/especialista"
- FR10.3: Registrar rota no DynamoDB para auditoria

### FR11 — Integração CRM HubSpot
O agente deve integrar com CRM via MCP demo + Private App REST.

**Sub-requisitos:**
- FR11.1: MCP HubSpot numa demonstração única ao vivo (usando MCP auth app + MCP Inspector)
- FR11.2: CRM-adapter (SQS→Lambda) com simulado default + Private App REST
- FR11.3: Sincronizar lead qualificado e status da esteira Kanban

### FR12 — Ingestão de Contato
O agente deve capturar contatos que chegam por e-mail/portais (cenário C7).

**Sub-requisitos:**
- FR12.1: Receber e-mails dos portais via Amazon SES
- FR12.2: Extrair nome, e-mail, telefone automaticamente
- FR12.3: Abrir sessão no bot enviando 1ª mensagem como o lead

## Requisitos Não Funcionais

### NFR1 — Performance
- **NFR1.1**: Primeira resposta < 10s (alinhado com KPIs do PRD, resolvendo inconsistência interna NF-03 < 4s)
- **NFR1.2**: Atendimento simultâneo sem fila

### NFR2 — Segurança (LGPD)
- **NFR2.1**: PII mascarada antes do envio ao LLM (nomes, telefone, e-mail, CNPJ)
- **NFR2.2**: Minimização de dados — PII real fica em registro criptografado (KMS), LLM recebe apenas placeholders
- **NFR2.3**: Registro de consentimento contextualizado na primeira mensagem
- **NFR2.4**: Trilha de auditoria de dados de leads
- **NFR2.5**: Retenção limitada (TTL 90 dias para conversas de leads frios)

### NFR3 — Privacidade de Dados
- **NFR3.1**: PII mascarada antes do envio ao provedor LLM (independente do provedor)
- **NFR3.2**: Consentimento registrado
- **NFR3.3**: Retenção limitada (TTL)

### NFR4 — Confiabilidade
- **NFR4.1**: Componentes serverless com DLQ
- **NFR4.2**: Retries no webhook

### NFR5 — Observabilidade
- **NFR5.1**: Logs estruturados (CloudWatch)
- **NFR5.2**: Traços distribuídos
- **NFR5.3**: Métricas de negócio

### NFR6 — Custo
- **NFR6.1**: Custo baixo (~R$ 15/mês via OpenRouter Claude 3.5 Haiku)
- **NFR6.2**: Limite de tokens e monitoramento de custo

### NFR7 — Segurança de Modelo
- **NFR7.1**: Guardrails/denied topics
- **NFR7.2**: Detecção de prompt injection
- **NFR7.3**: Evasão de PII

### NFR8 — Escalabilidade
- **NFR8.1**: Escala horizontal automática (Lambda/API GW/EventBridge)

### NFR9 — Infra como Código (IaC)
- **NFR9.1**: Toda infra em Terraform (um `.tf` por serviço)
- **NFR9.2**: Deploy via `start.sh` (build → zip → `terraform apply`)
- **NFR9.3**: Teardown via `stop.sh` (`terraform destroy`)
- **NFR9.4**: Ambiente recriável de ponta a ponta

## Restrições

### Restrições Técnicas
- **RT1**: Stack: Python (LangGraph/LangChain, FastAPI, Streamlit)
- **RT2**: AWS serverless: Lambda, DynamoDB, S3, EventBridge, API Gateway, CloudWatch
- **RT3**: LLM via OpenRouter (Claude 3.5 Haiku via LiteLLM)
- **RT4**: Conta AWS dedicada/novo para a POC
- **RT5**: Canal de atendimento: Telegram (POC); WhatsApp no roadmap
- **RT6**: Dashboard: Streamlit Community Cloud (grátis), login via Cognito

### Restrições de Negócio
- **RB1**: Prazo: 12 de outubro (preferencialmente antes para gravação do vídeo de apresentação)
- **RB2**: Público-alvo principal: Empresas B2B (PMEs a multinacionais) buscando espaço corporativo
- **RB3**: Público-alvo secundário: Investidor PF (parcela menor) buscando salas para renda
- **RB4**: Integrações: HubSpot (MCP demo + Private App REST), base imóveis simulada (JSON + S3)
- **RB5**: Sem requisito regulatório específico além do bom senso de segurança

### Restrições Organizacionais
- **RO1**: Sem bloqueadores organizacionais — projeto novo, sem sistemas legados críticos
- **RO2**: Decisões de direção do produto são do time de projeto/equipe do hackathon
- **RO3**: Stakeholder de negócio para validação é o gestor da W Levitt

## Priorização (Must-have vs Nice-to-have)

### Must-have (Obrigatórios para entrega)
- Núcleo conversacional (FR1, FR2, FR3, FR4, FR5, FR6)
- Detecção de anomalias (FR9)
- CRM HubSpot (FR11)
- Voice (FR1.2)

### Nice-to-have (Desejáveis se couberem no prazo)
- Roleta de distribuição + esteira Kanban (FR10, FR7.3)
- Follow-up automático (FR8)
- Ingestão de e-mail (FR12)

## Assumptions

- **A1**: O PRD preliminar é a fonte principal de requisitos, refinado pelas decisões do AI-DLC
- **A2**: A POC deve ser funcional e demonstrável, não produto final
- **A3**: A prioridade é entregar todos os diferenciais do desafio (RAG, memória, multiagentes, etc.)
- **A4**: A arquitetura serverless AWS é um requisito do desafio
- **A5**: Dados são sintéticos e calibrados por fontes públicas (FipeZAP, Secovi-SP, GeoSampa)
- **A6**: Telegram é usado na POC por custo zero; arquitetura mantém canal agnóstico para WhatsApp futuro

## Fora de Escopo

- Integração nativa real com CRM (Kenlo/CS) — será simulada via API local
- Pagamento online de propostas
- Voice AI em produção (somente demo futura)
- WhatsApp nativo (roadmap pós-POC)
- Reporte formal periódico (semanal/mensal) — dashboard tempo real cobre o agregado

## Open Questions

- OQ1: Confirmação final da estratégia de priorização (must-have vs nice-to-have) para o prazo
- OQ2: Validação dos KPIs de sucesso com o gestor da W Levitt
- OQ3: Definição precisa das regras da roleta de distribuição (quais regras configuráveis)

## Traceability

- **Fonte**: PRD preliminar `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`
- **Enunciado**: `documentos/POSTECH - Hacka Agente_SDR_Imobiliario - Fase 5.md`
- **Mentoria**: `documentos/transcricao-mentoria-21.md`
- **Decisões AI-DLC**: Refinamentos de priorização, público-alvo expandido, métricas consistentes