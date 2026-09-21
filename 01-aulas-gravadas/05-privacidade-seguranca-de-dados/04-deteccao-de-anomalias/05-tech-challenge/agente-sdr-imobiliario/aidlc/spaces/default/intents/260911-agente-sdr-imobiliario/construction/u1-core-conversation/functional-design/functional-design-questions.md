# Functional Design Questions — U1: Core Conversation

## Q1: Workflow de Conversação (SalesFlow)

Como o fluxo conversacional deve ser estruturado?

- **Estados**: Saudação → Elicitação → Intenção → Qualificação → Recomendação → Agendamento → Follow-up → Handoff
- **Transições**: Baseadas em respostas do lead e decisões do SDR Agent
- **Contexto**: Mantido entre estados no DynamoDB

**Resposta do PRD (§7.2)**:
- **Engine de fluxo**: sales-flow (LangGraph) — grafo de estados com nós para saudação, elicitação, intenção, qualificação, recomendação, agendamento, follow-up e handoff
- **Framework**: LangGraph — framework de grafo de estados/agentes (nós e transições) que orquestra o sales-flow

[Answer]: A

---

## Q2: Modelo de Domínio (Lead)

Quais atributos e relacionamentos da entidade Lead?

- **Atributos**: lead_id, telegram_user_id, score, urgência, intenção, budget, área, região, deadline, nº pessoas, decisor, status, created_at, updated_at
- **Relacionamentos**: Lead tem muitas Conversations, Lead tem muitos Appointments, Lead tem muitos Handoffs

**Resposta do PRD (§7.1 + Domain Design)**:
- **Lead**: lead_id, telegram_user_id, score, urgency, intent, budget, area, region, deadline, people_count, decision_maker, status, created_at, updated_at
- **Conversation**: session_id, lead_id, messages, context, current_state, pii_masked, consent_recorded, created_at, ttl
- **Relacionamento**: each Conversation belongs to one Lead

[Answer]: A

---

## Q3: Regras de Negócio (PII Masking)

Como PII masking deve ser implementado?

- **Extrair**: Nome, e-mail, telefone, CNPJ do texto
- **Persistir**: PII real no DynamoDB criptografado (KMS)
- **Mascarar**: Substituir PII por placeholders antes de enviar ao LLM
- **Validar**: Saída do LLM contra vazamento de PII (regex)

**Resposta do PRD (§7.2, §8.8)**:
- **Security-layer**: máscara de PII antes do LLM (nomes, telefone, e-mail, CNPJ), registro de consentimento, guardrails de tópicos, validação de entrada (prompt-injection check)
- **PII persistence**: extrai e persiste a PII real no DynamoDB criptografado (KMS) — o LLM recebe só o bloco mascarado (placeholders) + atributos estruturados, sem o valor real em texto livre
- **Decisão**: masking e retenção mínima garantidos em código (OpenRouter POC); migração para Bedrock em produção quando residência formal de dados for necessária

[Answer]: A

---

## Q4: Regras de Negócio (Intenção Detection)

Como intenção deve ser detectada?

- **Classificação**: Compra/locação/investimento
- **Precisão**: ≥ 85% (conforme requirements)
- **Método**: LLM classification com few-shot prompts
- **Fallback**: Perguntar confirmação se baixa confiança

**Resposta do PRD (§7.2, §2.4)**:
- **Lead-qualifier**: extrai estrutura (JSON) e classifica intenção + urgência + budget; grava na ficha do lead
- **FR-08**: Identificar intenção (compra/aluguel/investimento) — Classificador de intenção (LLM + heurísticas) no início do fluxo
- **Score/pesos**: estratégia de somar pontos durante a conversa até o lead "desaguar" para o corretor — implementada no lead-qualifier (código) com explicação no prompt

[Answer]: A

---

## Q5: Regras de Negócio (Qualificação e Scoring)

Como qualificação e scoring devem ser calculados?

- **Score**: Prontidão + urgência (ponderação)
- **Threshold**: Score ≥ X qualifica para handoff
- **Inputs**: Metragem, região, orçamento, prazo, nº pessoas, decisor
- **Explainability**: Score explicável (fatores contribuintes)

**Resposta do PRD (§2.4, §7.2)**:
- **FR-04**: Qualificação de leads — Questionário adaptativo + classificação (compra/aluguel/investimento) + score
- **Score/pesos**: estratégia de somar pontos durante a conversa até o lead "desaguar" para o corretor — implementada no lead-qualifier (código) com explicação no prompt
- **FR-37**: Score explicável (explicação da pontuação em linguagem natural)

[Answer]: A

---

## Q6: Integração RAG (PropertiesRAG)

Como RAG deve ser integrado ao fluxo conversacional?

- **Trigger**: Lead pergunta por imóveis ou score atinge threshold
- **Busca**: Top-k imóveis compatíveis com filtros
- **Constraint**: Nunca inventar imóveis que não estão na base
- **Contexto**: Filtros do lead (metragem, região, orçamento)

**Resposta do PRD (§7.2, §7.3)**:
- **RAG — properties-rag**: vetoriza a base sintética de imóveis (S3) + embeddings; na POC usa FAISS local (índice ~200 docs carrega em memória lambda) para custo zero
- **SDR Agent**: geração de resposta via OpenRouter usando LiteLLM; tools de RAG lookup e scheduling
- **Constraint**: Quando score ≥ limite OU o lead pede, o agente propõe até 3 opções da base (RAG)

[Answer]: A

---

## Q7: Error Handling

Como erros devem ser tratados?

- **Secret inválido**: 401, Telegram deve reenviar
- **Payload inválido**: 400, Telegram deve corrigir
- **LLM timeout**: Retry com fallback
- **DynamoDB timeout**: Retry automático
- **PII leakage**: Bloquear resposta, log erro

**Resposta do PRD (§7.5, §7.1)**:
- **POST /webhook**: 200 se aceito, 401 se secret inválido, 400 se payload inválido
- **SQS**: DLQ para falhas, retry automático
- **DynamoDB**: Retry automático, exponential backoff
- **PII**: Masking antes do LLM, validação de saída contra vazamento de PII (regex)

[Answer]: A

---

## Summary Confirmation

**Resumo consolidado das respostas (baseado no PRD §7.1-7.3, §8.8-8.10)**:
- **Q1 (Workflow conversacional)**: A - LangGraph com nós para cada estado, transições baseadas em LLM decisions
- **Q2 (Modelo de domínio Lead)**: A - Modelo definido em Domain Design (conforme components.md)
- **Q3 (Regras de PII masking)**: A - SecurityLayer como módulo interno de ConversationRouter (conforme ADR-008)
- **Q4 (Intenção detection)**: A - LLM classification com few-shot prompts + fallback
- **Q5 (Qualificação e scoring)**: A - Scoring ponderado com explainability (conforme requirements)
- **Q6 (Integração RAG)**: A - SDR Agent chama PropertiesRAG como tool quando apropriado
- **Q7 (Error handling)**: A - Conforme contract-summary (OpenAPI + DLQ + retry)