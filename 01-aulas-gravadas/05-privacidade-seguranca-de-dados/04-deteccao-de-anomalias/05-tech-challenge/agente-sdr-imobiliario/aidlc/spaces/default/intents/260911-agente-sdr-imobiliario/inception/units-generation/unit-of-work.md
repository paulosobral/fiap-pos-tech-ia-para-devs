# Unit of Work — Agente SDR Imobiliário B2B

> Estágio Units Generation (Inception). Fonte: components.md, decisions.md, requirements.md, stories.md, units-generation-questions (Q1-Q5).

---

## Unit Definitions

| Unit ID | Directory | Description | Kind | Deployment Model | Complexity |
|---------|-----------|-------------|------|------------------|------------|
| U1 | u1-core-conversation | Núcleo síncrono — ConversationRouter com módulos internos (SalesFlow, SecurityLayer, SDR Agent, LeadQualifier, PropertiesRAG, Scheduler, Handoff, LeadRouter) | service | Monolithic (deploy com outras unidades) | XL |
| U2 | u2-async-voice | Adaptador de voz assíncrono — VoiceAdapter (SQS → transcrição faster-whisper) | service | Monolithic (deploy com outras unidades) | M |
| U3 | u3-async-crm | Adaptador CRM assíncrono — CRMAdapter (SQS → MCP HubSpot/Kenlo/Facilita) | service | Monolithic (deploy com outras unidades) | M |
| U4 | u4-async-ingest | Ingestão de contato assíncrono — ContactIngest (SES → Telegram) | service | Monolithic (deploy com outras unidades) | M |
| U5 | u5-anomaly | Detecção de anomalias — AnomalyDetector (EventBridge job diário) | service | Monolithic (deploy com outras unidades) | L |
| U6 | u6-followup | Follow-up automático — Followup (EventBridge + Step Functions) | service | Monolithic (deploy com outras unidades) | M |
| U7 | u7-dashboard | Dashboard — DashAPI + Dashboard (Streamlit Community Cloud) | ui + service | Monolithic (deploy com outras unidades) | L |

---

## Unit Responsibilities

### U1: Core Conversation

**Responsibilities:**
- Receber webhook do Telegram e autenticar
- Gerenciar sessões de conversa (criação, recuperação, persistência)
- Orquestrar SalesFlow (LangGraph)
- Aplicar SecurityLayer (PII masking, guardrails)
- Gerar respostas humanizadas via SDR Agent (OpenRouter)
- Extrair estrutura e classificar intenção via LeadQualifier
- Buscar imóveis via PropertiesRAG (FAISS local)
- Validar e agendar visitas via Scheduler
- Gerar handoff inteligente via Handoff
- Aplicar roleta de distribuição via LeadRouter
- Enfileira áudio para SQS
- Enfileira leads qualificados para SQS

**Implementation Notes:**
- Contém todos os módulos síncronos internos (conforme ADR-001)
- Usa DynamoDB para sessões e leads
- Usa S3 para carregar índice FAISS
- Usa OpenRouter para LLM
- Envia mensagens ao Telegram
- Usa Secrets Manager para tokens

**Constraints:**
- Deve responder em < 10s (just-in-time da mentoria)
- Deve aplicar PII masking antes de enviar ao LLM
- Deve manter contexto conversacional entre mensagens

---

### U2: Async Voice

**Responsibilities:**
- Receber mensagens de áudio via SQS
- Baixar arquivo via Telegram getFile
- Converter para WAV (ffmpeg)
- Transcrever com faster-whisper PT-BR
- Enviar texto transcrito de volta para ConversationRouter

**Implementation Notes:**
- Lambda própria acionada por SQS
- Usa faster-whisper layer
- Processamento assíncrono (não bloqueia resposta ao lead)

**Constraints:**
- Transcrição deve acontecer em < 15s para áudios < 30s
- Deve ter fallback para áudios inválidos

---

### U3: Async CRM

**Responsibilities:**
- Receber leads qualificados via SQS
- Ler/gravar leads no CRM via MCP
- Sincronizar status da esteira Kanban
- Devolver status para o fluxo

**Implementation Notes:**
- Lambda própria acionada por SQS
- Usa MCP server (HubSpot ou Kenlo/Facilita)
- Na POC roda contra CRM simulado (CSV/Excel)

**Constraints:**
- Deve ser desacoplado do fluxo do agente
- Deve ter DLQ para falhas

---

### U4: Async Ingest

**Responsibilities:**
- Receber e-mails dos portais via SES
- Extrair nome, e-mail, telefone
- Abrir sessão no bot automaticamente
- Enviar primeira mensagem como o lead

**Implementation Notes:**
- Lambda própria acionada por SES
- Integra com ConversationRouter para abrir sessão

**Constraints:**
- Deve extrair dados corretamente de e-mails
- Deve evitar spam/duplicatas

---

### U5: Anomaly

**Responsibilities:**
- Job diário (EventBridge)
- Extrair features por conversa (volume, comprimento, sentimento, horários atípicos)
- Aplicar Isolation Forest + PCA + Autoencoder
- Emitir alerta no dashboard
- Restringir agendamento para leads suspeitos

**Implementation Notes:**
- Lambda própria acionada por EventBridge
- Lê conversas do DynamoDB
- Emite alertas para dashboard

**Constraints:**
- Deve rodar diariamente
- Deve ter falso-positivo documentado para validação

---

### U6: Followup

**Responsibilities:**
- Regras de cadência via EventBridge
- Wait states via Step Functions (dia 2, 5, 9)
- Retomar lead parado com contexto
- Respeitar janela de silêncio

**Implementation Notes:**
- Lambda própria acionada por EventBridge + Step Functions
- Lê contexto do DynamoDB
- Envia mensagem via Telegram

**Constraints:**
- Deve respeitar janela de silêncio
- Deve manter contexto da última conversa

---

### U7: Dashboard

**Responsibilities:**
- DashAPI: Agregar KPIs (DynamoDB + CloudWatch)
- Dashboard: UI Streamlit (1 página) com KPIs, esteira Kanban, alertas
- Login via Cognito

**Implementation Notes:**
- DashAPI é Lambda própria
- Dashboard é app Streamlit no Community Cloud
- Consome GET /api/kpis

**Constraints:**
- Dashboard não guarda dados locais
- Deve ter login seguro (Cognito)

---

## Deployment Model

**Monolithic deploy (Q5-A):**
- Todas as unidades são deployadas juntas via `start.sh`
- `start.sh` builda todas as Lambdas e roda `terraform apply`
- `stop.sh` roda `terraform destroy`
- Ambiente recriável de ponta a ponta

**Justification:**
- Simplicidade para POC
- Alinhado com prática aprovada (manual deployment via start.sh/stop.sh)
- Menos overhead de coordenação
- Walking skeleton rápido