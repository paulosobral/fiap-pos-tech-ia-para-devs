# External Dependency Map — Agente SDR Imobiliário B2B

> Estágio Delivery Planning (Inception). Fonte: components.md, contract-summary.md, delivery-planning-questions (Q5-A: Mockar tudo).

---

## External Dependencies Overview

| Dependency | Type | Used By | Bolt | Mock/Real | Status | Notes |
|------------|------|---------|------|-----------|--------|-------|
| Telegram Bot API | Third-party API | U1 (Core Conversation) | B1, B2 | Mock → Real | B1 mock, B2 real | Webhook e envio de mensagens |
| OpenRouter API | Third-party API | U1 (SDR Agent) | B1, B2 | Mock → Real | B1 mock, B2 real | LLM (Claude 3.5 Haiku via LiteLLM) |
| DynamoDB | AWS Service | U1, U2, U3, U4, U5, U6, U7 | All | Real | Ready | Sessões, leads, conversas, alertas |
| SQS | AWS Service | U1, U2, U3 | B1, B3, B4 | Real | Ready | Filas de áudio e CRM |
| S3 | AWS Service | U1 (PropertiesRAG) | B2 | Real | Ready | Índice FAISS de imóveis |
| EventBridge | AWS Service | U5, U6 | B6, B7 | Real | Ready | Job diário e cadências |
| Step Functions | AWS Service | U6 | B7 | Real | Ready | Wait states de follow-up |
| Lambda | AWS Service | All | All | Real | Ready | Runtime de todas as unidades |
| API Gateway | AWS Service | U1, U7 | B1, B8 | Real | Ready | POST /webhook, GET /api/kpis |
| Cognito | AWS Service | U7 | B8 | Real | Ready | Login do dashboard |
| Secrets Manager | AWS Service | U1 | B1, B2 | Real | Ready | Token do bot e chaves de API |
| KMS | AWS Service | U1 (SecurityLayer) | B2 | Real | Ready | Criptografia de PII |
| SES | AWS Service | U4 | B5 | Real | Ready | Receber e-mails de portais |
| CloudWatch | AWS Service | U7 (DashAPI) | B8 | Real | Ready | Métricas de operação |
| faster-whisper | Library/Model | U2 (VoiceAdapter) | B3 | Real | Ready | Transcrição PT-BR |
| ffmpeg | Tool | U2 (VoiceAdapter) | B3 | Real | Ready | Conversão áudio → WAV |
| LiteLLM | Library | U1 (SDR Agent) | B2 | Real | Ready | Cliente abstraído para OpenRouter |
| LangGraph | Library | U1 (SalesFlow) | B2 | Real | Ready | Engine de fluxo conversacional |
| LangChain | Library | U1 (SDR Agent) | B2 | Real | Ready | Framework LLM |
| FAISS | Library | U1 (PropertiesRAG) | B2 | Real | Ready | Vetor search local |
| Streamlit | Library | U7 (Dashboard) | B8 | Real | Ready | UI dashboard |
| MCP Server | Library/Protocol | U3 (CRMAdapter) | B4 | Real → Mock | B4 mock (CSV/Excel) | HubSpot/Kenlo/Facilita |

---

## Dependency-by-Bolt Breakdown

### B1: Walking Skeleton

**Dependencies**: Telegram Bot API (mock), OpenRouter API (mock), DynamoDB (real), SQS (real), Lambda (real), API Gateway (real), Secrets Manager (real)

**Mock Strategy**:
- Telegram Bot API: Mock que simula webhook e respostas
- OpenRouter API: Mock que simula respostas do LLM
- AWS services: Reais (devem funcionar em B1)

**Justification**: Q5-A (Mockar tudo) — reduz risco de dependências externas críticas em walking skeleton

---

### B2: Core Conversation Complete

**Dependencies**: Telegram Bot API (real), OpenRouter API (real), DynamoDB (real), SQS (real), S3 (real), Lambda (real), API Gateway (real), Secrets Manager (real), KMS (real), LiteLLM (real), LangGraph (real), LangChain (real), FAISS (real)

**Real Integration Strategy**:
- Telegram Bot API: Webhook real, token configurado
- OpenRouter API: Real integration com Claude 3.5 Haiku
- AWS services: Reais
- Libraries: Reais

**Justification**: B2 valida integração real com dependências críticas

---

### B3: Async Voice

**Dependencies**: Telegram Bot API (real), SQS (real), Lambda (real), faster-whisper (real), ffmpeg (real)

**Real Integration Strategy**:
- Telegram Bot API: getFile real
- SQS: Fila real
- faster-whisper: Modelo PT-BR real
- ffmpeg: Tool real

**Justification**: Transcrição de voz requer integração real

---

### B4: Async CRM

**Dependencies**: SQS (real), Lambda (real), MCP Server (mock)

**Mock Strategy**:
- MCP Server: Mock que simula HubSpot (CSV/Excel)
- SQS: Real

**Justification**: Q5-A (Mockar tudo) — CRM simulado para POC

---

### B5: Async Ingest

**Dependencies**: SES (real), DynamoDB (real), Lambda (real), Telegram Bot API (real)

**Real Integration Strategy**:
- SES: Receber e-mails real
- DynamoDB: Real
- Telegram Bot API: Enviar primeira mensagem real

**Justification**: Ingestão de e-mail requer integração real

---

### B6: Anomaly

**Dependencies**: EventBridge (real), DynamoDB (real), Lambda (real)

**Real Integration Strategy**:
- EventBridge: Job diário real
- DynamoDB: Real
- Lambda: Real

**Justification**: Job diário requer integração real

---

### B7: Followup

**Dependencies**: EventBridge (real), Step Functions (real), DynamoDB (real), Lambda (real), Telegram Bot API (real)

**Real Integration Strategy**:
- EventBridge: Cadências reais
- Step Functions: Wait states reais
- DynamoDB: Real
- Telegram Bot API: Enviar mensagem real

**Justification**: Follow-up requer integração real

---

### B8: Dashboard

**Dependencies**: DynamoDB (real), CloudWatch (real), Lambda (real), API Gateway (real), Cognito (real), Streamlit (real)

**Real Integration Strategy**:
- DynamoDB: Real
- CloudWatch: Real
- Cognito: Real
- Streamlit: Real

**Justification**: Dashboard requer integração real

---

## Dependency Acquisition Strategy

### AWS Services

**Provisioning**: Terraform (apigw.tf, lambda.tf, dynamodb.tf, sqs.tf, s3.tf, eventbridge.tf, stepfunctions.tf, cognito.tf, ses.tf, secretsmanager.tf, kms.tf, cloudwatch.tf)

**Justification**: PRD §7.1 define Terraform como IaC, prática aprovada em Requirements Analysis

---

### Third-party APIs

**Telegram Bot API**:
- Token configurado via Secrets Manager
- Webhook configurado via API Gateway

**OpenRouter API**:
- Key configurada via Secrets Manager
- Cliente via LiteLLM

---

### Libraries/Models

**Python Libraries**:
- LiteLLM, LangGraph, LangChain, FAISS, Streamlit
- Instalados via requirements.txt

**faster-whisper**:
- Layer Lambda customizada ou instalado via pip

**ffmpeg**:
- Tool instalado no ambiente Lambda (layer ou image)

---

### MCP Server

**HubSpot**:
- MCP server configurado para demo única ao vivo
- POC roda contra CRM simulado (CSV/Excel)

---

## Dependency Risk Matrix

| Dependency | Failure Impact | Recovery Time | Mitigation |
|------------|----------------|---------------|------------|
| Telegram Bot API | High (no webhook) | Low (switch to WhatsApp) | Documentação de fallback |
| OpenRouter API | High (no LLM) | Low (switch to Bedrock) | LiteLLM abstraction |
| DynamoDB | High (no state) | Medium (backup/restore) | Regular backups |
| SQS | Medium (no async) | Low (retry queue) | DLQ + retry |
| S3 | Medium (no RAG) | Low (rebuild index) | Backup de índice |
| EventBridge | Low (no scheduled jobs) | Low (manual trigger) | Manual override |
| Step Functions | Low (no follow-up) | Low (manual follow-up) | Manual follow-up |
| Cognito | Medium (no dashboard login) | Low (bypass auth) | Dev mode bypass |
| Secrets Manager | High (no keys) | Medium (rotate keys) | Key rotation |
| KMS | High (no encryption) | Medium (restore keys) | Key backup |
| SES | Low (no email ingest) | Low (manual ingest) | Manual ingest |
| CloudWatch | Low (no metrics) | Low (manual logs) | Manual logs |
| faster-whisper | Medium (no voice) | Low (manual transcription) | Manual transcription |
| ffmpeg | Medium (no voice) | Low (manual conversion) | Manual conversion |
| LiteLLM | High (no LLM) | Low (switch provider) | Switch provider |
| LangGraph | High (no conversation flow) | Medium (manual flow) | Manual flow |
| LangChain | High (no LLM framework) | Medium (manual LLM) | Manual LLM |
| FAISS | Medium (no RAG) | Low (rebuild index) | Rebuild index |
| Streamlit | High (no dashboard) | Low (alternative UI) | Alternative UI |
| MCP Server | Low (no CRM) | Low (manual CRM) | Manual CRM |