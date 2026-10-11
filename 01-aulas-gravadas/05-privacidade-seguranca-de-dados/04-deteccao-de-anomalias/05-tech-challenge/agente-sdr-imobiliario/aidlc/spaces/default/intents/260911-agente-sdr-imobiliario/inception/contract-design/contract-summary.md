# Contract Summary — Agente SDR Imobiliário B2B

> Estágio Contract Design (Inception). Fonte: unit-of-work.md, unit-of-work-dependency.md, components.md, PRD §7.5 (apigw-openapi.yaml), contract-design-questions (Q1-Q5).

---

## Contracts Table

| Boundary | Type | Provider Unit | Consumer Unit | Integration Mechanism | Contract Owner | Versioning | Error/Timeout/Retry |
|----------|------|---------------|----------------|----------------------|----------------|------------|---------------------|
| POST /webhook | Public API | U1 (Core Conversation) | Telegram Bot API | HTTP/REST (aws_proxy) | U1 | 1.0.0 (POC) | 200 aceito, 401 secret inválido, 400 payload inválido |
| GET /api/kpis | Public API | U7 (DashAPI) | Dashboard (Streamlit) | HTTP/REST (aws_proxy) | U7 | 1.0.0 (POC) | 200 KPIs, 401 não autenticado, 500 erro |
| SQS voice | Inter-unit | U1 (producer) | U2 (consumer) | SQS (async message) | U1 | None (POC) | DLQ para falhas, retry automático |
| SQS CRM | Inter-unit | U1 (producer) | U3 (consumer) | SQS (async message) | U1 | None (POC) | DLQ para falhas, retry automático |
| DynamoDB (sessões) | Shared Data | U1 (owner) | U2, U3, U4, U5, U6, U7 (readers) | DynamoDB (shared schema) | U1 | None (POC) | Retry automático, exponential backoff |
| DynamoDB (conversas) | Shared Data | U1 (owner) | U5 (reader) | DynamoDB (shared schema) | U1 | None (POC) | Retry automático, exponential backoff |
| DynamoDB (alertas) | Shared Data | U5 (owner) | U7 (reader) | DynamoDB (shared schema) | U5 | None (POC) | Retry automático, exponential backoff |

---

## Contract 1: POST /webhook (Public API)

**Boundary Type**: Public API  
**Provider Unit**: U1 (Core Conversation)  
**Consumer**: Telegram Bot API  
**Integration Mechanism**: HTTP/REST (aws_proxy via API Gateway)  
**Contract Owner**: U1

### OpenAPI 3.0 Spec

```yaml
openapi: 3.0.3
info:
  title: W Levitt — SDR API
  version: 1.0.0
  description: >
    Única superfície HTTP da POC. POST /webhook recebe o update do Telegram e
    responde 200 imediato (trabalho lento vai para SQS — ver §7.1 do PRD).
    Deployado pelo apigw.tf via aws_api_gateway_rest_api.body.
servers:
  - url: https://api.wlevitt.app
paths:
  /webhook:
    post:
      summary: Webhook do Telegram
      description: >
        Roteia o update para a Lambda conversation-router (proxy).
        Autentica pelo secret token do bot (header enviado pelo Telegram).
      security:
        - TelegramSecret: []
      parameters:
        - in: header
          name: X-Telegram-Bot-Api-Secret-Token
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/TelegramUpdate'
      responses:
        '200':
          description: Update aceito (processado síncrono ou enfileirado)
        '401':
          description: Secret inválido
        '400':
          description: Payload inválido
      x-amazon-apigateway-integration:
        type: aws_proxy
        httpMethod: POST
        uri: ${ROUTER_INVOKE_ARN}
        passthroughBehavior: when_no_match
components:
  securitySchemes:
    TelegramSecret:
      type: apiKey
      in: header
      name: X-Telegram-Bot-Api-Secret-Token
  schemas:
    TelegramUpdate:
      type: object
      properties:
        update_id:
          type: integer
        message:
          type: object
          properties:
            message_id:
              type: integer
            from:
              type: object
            chat:
              type: object
            date:
              type: integer
            text:
              type: string
            voice:
              type: object
            document:
              type: object
```

### Error/Timeout/Retry Behaviour

- **200**: Update aceito (processado síncrono ou enfileirado)
- **401**: Secret inválido — Telegram deve reenviar com secret correto
- **400**: Payload inválido — Telegram deve corrigir payload
- **Timeout**: API Gateway timeout padrão (29s) — deve ser suficiente para processamento síncrono rápido
- **Retry**: Sem retry — Telegram retryará se não receber 200

---

## Contract 2: GET /api/kpis (Public API)

**Boundary Type**: Public API  
**Provider Unit**: U7 (DashAPI)  
**Consumer**: Dashboard (Streamlit)  
**Integration Mechanism**: HTTP/REST (aws_proxy via API Gateway)  
**Contract Owner**: U7

### OpenAPI 3.0 Spec

```yaml
paths:
  /api/kpis:
    get:
      summary: KPIs do dashboard
      description: >
        Agrega métricas de negócio (leads, qualificados, agendamentos) e
        operação (tempo de resposta, custo) do DynamoDB e CloudWatch.
      security:
        - CognitoAuthorizer: []
      responses:
        '200':
          description: JSON de KPIs
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/KPIs'
        '401':
          description: Não autenticado
        '500':
          description: Erro interno
      x-amazon-apigateway-integration:
        type: aws_proxy
        httpMethod: GET
        uri: ${DASHAPI_INVOKE_ARN}
        passthroughBehavior: when_no_match
components:
  securitySchemes:
    CognitoAuthorizer:
      type: http
      scheme: bearer
      bearerFormat: JWT
      x-amazon-apigateway-authorizer:
        type: cognito_user_pools
        providerARNs:
          - ${COGNITO_POOL_ARN}
  schemas:
    KPIs:
      type: object
      properties:
        leads_today:
          type: integer
        leads_week:
          type: integer
        response_time_p90:
          type: number
        qualification_rate:
          type: number
        scheduled_visits:
          type: integer
        anomalies_count:
          type: integer
        cost_monthly:
          type: number
```

### Error/Timeout/Retry Behaviour

- **200**: JSON de KPIs
- **401**: Não autenticado — Dashboard deve pedir login via Cognito
- **500**: Erro interno — Dashboard deve mostrar erro toast
- **Timeout**: API Gateway timeout padrão (29s) — deve ser suficiente para agregação
- **Retry**: Dashboard retryará automaticamente se erro

---

## Contract 3: SQS voice (Inter-unit)

**Boundary Type**: Inter-unit  
**Provider Unit**: U1 (producer)  
**Consumer Unit**: U2 (consumer)  
**Integration Mechanism**: SQS (async message)  
**Contract Owner**: U1

### Message Schema

```json
{
  "message_id": "uuid",
  "telegram_user_id": "integer",
  "voice_file_id": "string",
  "session_id": "string",
  "timestamp": "iso8601"
}
```

### Error/Timeout/Retry Behaviour

- **DLQ**: Mensagens com falha de processamento vão para DLQ
- **Retry**: SQS retry automático (configurável, padrão 3 tentativas)
- **Timeout**: Visibility timeout (configurável, padrão 30s)
- **Backoff**: Exponential backoff entre retries

---

## Contract 4: SQS CRM (Inter-unit)

**Boundary Type**: Inter-unit  
**Provider Unit**: U1 (producer)  
**Consumer Unit**: U3 (consumer)  
**Integration Mechanism**: SQS (async message)  
**Contract Owner**: U1

### Message Schema

```json
{
  "message_id": "uuid",
  "lead_id": "string",
  "lead_data": {
    "name": "string",
    "email": "string",
    "phone": "string",
    "score": "number",
    "urgency": "string",
    "intent": "string"
  },
  "session_id": "string",
  "timestamp": "iso8601"
}
```

### Error/Timeout/Retry Behaviour

- **DLQ**: Mensagens com falha de processamento vão para DLQ
- **Retry**: SQS retry automático (configurável, padrão 3 tentativas)
- **Timeout**: Visibility timeout (configurável, padrão 30s)
- **Backoff**: Exponential backoff entre retries

---

## Contract 5: DynamoDB (Shared Data - Sessões)

**Boundary Type**: Shared Data  
**Provider Unit**: U1 (owner)  
**Consumer Units**: U2, U3, U4, U5, U6, U7 (readers)  
**Integration Mechanism**: DynamoDB (shared schema)  
**Contract Owner**: U1

### Schema (Lead)

```json
{
  "lead_id": "string (PK)",
  "telegram_user_id": "integer",
  "score": "number",
  "urgency": "string",
  "intent": "string",
  "budget": "string",
  "area": "string",
  "region": "string",
  "deadline": "string",
  "people_count": "integer",
  "decision_maker": "string",
  "status": "string",
  "created_at": "iso8601",
  "updated_at": "iso8601"
}
```

### Schema (Conversation)

```json
{
  "session_id": "string (PK)",
  "lead_id": "string",
  "messages": "array",
  "context": "object",
  "current_state": "string",
  "pii_masked": "boolean",
  "consent_recorded": "boolean",
  "created_at": "iso8601",
  "ttl": "integer"
}
```

### Error/Timeout/Retry Behaviour

- **Retry**: DynamoDB retry automático (configurável, padrão 3 tentativas)
- **Timeout**: Request timeout (configurável, padrão 5s)
- **Backoff**: Exponential backoff entre retries
- **Throttling**: DynamoDB throttling tratado como erro, retornado

---

## Contract 6: DynamoDB (Shared Data - Conversas)

**Boundary Type**: Shared Data  
**Provider Unit**: U1 (owner)  
**Consumer Unit**: U5 (reader)  
**Integration Mechanism**: DynamoDB (shared schema)  
**Contract Owner**: U1

### Schema (Conversation)

```json
{
  "session_id": "string (PK)",
  "lead_id": "string",
  "messages": "array",
  "context": "object",
  "current_state": "string",
  "pii_masked": "boolean",
  "consent_recorded": "boolean",
  "created_at": "iso8601",
  "ttl": "integer"
}
```

### Error/Timeout/Retry Behaviour

- **Retry**: DynamoDB retry automático (configurável, padrão 3 tentativas)
- **Timeout**: Request timeout (configurável, padrão 5s)
- **Backoff**: Exponential backoff entre retries
- **Throttling**: DynamoDB throttling tratado como erro, retornado

---

## Contract 7: DynamoDB (Shared Data - Alertas)

**Boundary Type**: Shared Data  
**Provider Unit**: U5 (owner)  
**Consumer Unit**: U7 (reader)  
**Integration Mechanism**: DynamoDB (shared schema)  
**Contract Owner**: U5

### Schema (Anomaly)

```json
{
  "anomaly_id": "string (PK)",
  "lead_id": "string",
  "features": "object",
  "confidence": "number",
  "type": "string",
  "detected_at": "iso8601",
  "status": "string",
  "action_taken": "string"
}
```

### Error/Timeout/Retry Behaviour

- **Retry**: DynamoDB retry automático (configurável, padrão 3 tentativas)
- **Timeout**: Request timeout (configurável, padrão 5s)
- **Backoff**: Exponential backoff entre retries
- **Throttling**: DynamoDB throttling tratado como erro, retornado

---

## Versioning and Breaking-Change Policy

**POC Versioning (Q4-A):**
- Sem versionamento formal para POC
- OpenAPI 3.0: versão 1.0.0
- SQS schemas: sem versionamento (JSON schema fixo)
- DynamoDB schemas: sem versionamento (schema fixo)
- Breaking changes não são suportadas na POC

**Post-POC:**
- Introduzir versionamento semântico (major.minor.patch)
- Breaking changes requerem major version bump
- Deprecation notice mínimo 30 dias antes de breaking change

---

## Integration Points Summary

**Public APIs:**
- POST /webhook (U1 → Telegram)
- GET /api/kpis (U7 → Dashboard)

**Inter-unit (SQS):**
- U1 → U2 (voice transcription)
- U1 → U3 (CRM sync)

**Inter-unit (DynamoDB):**
- U1 (owner) → U2, U3, U4, U5, U6, U7 (readers of sessions/conversations)
- U5 (owner) → U7 (reader of alerts)