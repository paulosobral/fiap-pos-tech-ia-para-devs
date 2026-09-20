# Contract Design Questions — Agente SDR Imobiliário B2B

## Q1: Superfície de API Pública

Quais unidades expõem API pública consumida fora do sistema?

- **U1 (Core Conversation)**: Exibe POST /webhook para Telegram
- **U7 (Dashboard)**: Exibe GET /api/kpis para Streamlit

**Opções**:
- A) U1 e U7 expõem APIs públicas (conforme PRD §7.5)
- B) Apenas U1 expõe API pública
- C) Apenas U7 expõe API pública
- D) Nenhuma unidade expõe API pública

---

## Q2: Mecanismo de Integração por Limite

Qual mecanismo de integração para cada limite inter-unidade?

- **U1 → U2 (SQS)**: Mensagem assíncrona com payload JSON
- **U1 → U3 (SQS)**: Mensagem assíncrona com payload JSON
- **U1 → U5 (DynamoDB)**: Leitura compartilhada de conversas
- **U1 → U6 (DynamoDB)**: Leitura compartilhada de contexto
- **U1 → U7 (DynamoDB)**: Leitura compartilhada de métricas
- **U5 → U7 (DynamoDB)**: Leitura compartilhada de alertas

**Opções**:
- A) SQS para assíncrono, DynamoDB para shared data (conforme PRD §7.1)
- B) SQS para tudo
- C) HTTP/REST para tudo
- D) Outro (especifique)

---

## Q3: Propriedade de Contrato

Qual unidade é dona de cada contrato?

- **POST /webhook**: U1 (Core Conversation)
- **GET /api/kpis**: U7 (Dashboard via DashAPI)
- **SQS voice**: U1 (producer), U2 (consumer)
- **SQS CRM**: U1 (producer), U3 (consumer)
- **DynamoDB schema**: U1 (owner), outras unidades (readers)

**Opções**:
- A) API pública: dono é a unidade que expõe; SQS: dono é producer; DynamoDB: dono é U1
- B) API pública: dono é U1 para tudo; SQS: dono é consumer; DynamoDB: dono é U7
- C) Todos os contratos são dono de U1
- D) Outro (especifique)

---

## Q4: Versionamento e Breaking Changes

Qual política de versionamento?

- **OpenAPI 3.0**: Versão 1.0.0 sem breaking changes para POC
- **SQS schema**: Sem versionamento para POC (JSON schema fixo)
- **DynamoDB schema**: Sem versionamento para POC (schema fixo)

**Opções**:
- A) Sem versionamento para POC (simples)
- B) Versionamento semântico (major.minor.patch)
- C) Versionamento por data (YYYY-MM-DD)
- D) Outro (especifique)

---

## Q5: Comportamento de Erro, Timeout e Retry

Qual comportamento em cada limite?

- **POST /webhook**: 200 se aceito, 401 se secret inválido, 400 se payload inválido
- **GET /api/kpis**: 200 com KPIs, 401 se não autenticado, 500 se erro
- **SQS**: DLQ para falhas, retry automático
- **DynamoDB**: Retry automático, exponential backoff

**Opções**:
- A) Conforme PRD §7.5 (OpenAPI) + DLQ para SQS + retry DynamoDB
- B) Retry automático para tudo
- C) Retry manual para tudo
- D) Outro (especifique)