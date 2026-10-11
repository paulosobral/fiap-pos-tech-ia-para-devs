# Rules — U1: Core Conversation

> Unidade U1 (Core Conversation). Fonte: components.md, functional-design-questions, PRD §7.1-7.3, §8.8.

---

## Rules Source of Truth

```yaml
rules:
  - id: R1
    name: PII Masking
    category: Security
    priority: Critical
    description: Extrair e persistir PII real criptografado, mascarar antes do LLM
    trigger: Mensagem do lead contém PII (nome, e-mail, telefone, CNPJ)
    condition: Texto contém padrões de PII
    action:
      - Extrair PII via regex
      - Persistir PII real no DynamoDB criptografado (KMS)
      - Substituir PII por placeholders no texto
      - Validar saída do LLM contra vazamento de PII (regex)
    exception:
      - Bloquear resposta se PII vazado
      - Log erro para auditoria

  - id: R2
    name: Intent Detection
    category: Business
    priority: High
    description: Classificar intenção (compra/locação/investimento) com ≥ 85% de precisão
    trigger: Início da conversa (primeiras 3 trocas)
    condition: Lead fornece informações sobre uso
    action:
      - Classificar intenção via LLM com few-shot prompts
      - Se confiança < threshold, pedir confirmação
      - Gravar intenção no Lead
    exception:
      - Perguntar "Você está buscando compra, locação ou investimento?"
      - Não prosseguir até intenção confirmada

  - id: R3
    name: Lead Qualification
    category: Business
    priority: High
    description: Calcular score de prontidão com explainability
    trigger: Lead fornece informações (metragem, região, orçamento, prazo, nº pessoas, decisor)
    condition: Informações qualificadas coletadas
    action:
      - Calcular score ponderado (prontidão + urgência)
      - Explicar score em linguagem natural
      - Gravar score e fatores no Lead
    exception:
      - Se informações incompletas, perguntar faltantes
      - Não qualificar se score < threshold

  - id: R4
    name: RAG Trigger
    category: Business
    priority: Medium
    description: Buscar imóveis via RAG quando score ≥ threshold ou lead pede
    trigger: Score ≥ threshold OU lead pergunta por imóveis
    condition: Filtros disponíveis (metragem, região, orçamento)
    action:
      - SDR Agent chama PropertiesRAG como tool
      - Buscar top-k imóveis compatíveis
      - Nunca inventar imóveis que não estão na base
    exception:
      - Se nenhum imóvel encontrado, explicar motivo
      - Sugerir ajustar filtros

  - id: R5
    name: Handoff Trigger
    category: Business
    priority: High
    description: Gerar handoff quando lead qualificado
    trigger: Score ≥ threshold OU lead solicita
    condition: Lead qualificado
    action:
      - LeadRouter aplica regras de distribuição
      - Handoff gera resumo Markdown
      - Desmascarar PII apenas no destino
      - Enviar resumo ao corretor
    exception:
      - Se nenhum corretor disponível, notificar gestor

  - id: R6
    name: Consent Recording
    category: Legal
    priority: Critical
    description: Registrar consentimento LGPD na primeira mensagem
    trigger: Lead envia `/start` ou primeira mensagem
    condition: Primeira interação
    action:
      - Apresentar consentimento contextualizado
      - Registrar consentimento na Conversation
      - Se recusado, não coletar dados, encerrar educadamente
    exception:
      - Sem exceção — consentimento é obrigatório

  - id: R7
    name: Guardrails
    category: Security
    priority: High
    description: Aplicar guardrails de tópicos negados e validação de entrada
    trigger: Mensagem do lead
    condition: Conteúdo viola guardrails
    action:
      - Bloquear conteúdo proibido
      - Responder com mensagem de fallback
      - Log violação para auditoria
    exception:
      - Sem exceção — guardrails são obrigatórios

  - id: R8
    name: Session Management
    category: Technical
    priority: Medium
    description: Gerenciar sessão com TTL de 90 dias
    trigger: Lead envia mensagem
    condition: Sessão expirada ou não existe
    action:
      - Criar nova sessão se não existe
      - Recuperar sessão existente se não expirada
      - Expirar sessão após 90 dias (DynamoDB TTL)
    exception:
      - Se sessão expirada, recriar e explicar contexto perdido
```

---

## Rules Summary Table

| Rule ID | Rule Name | Category | Priority | Trigger | Action |
|---------|-----------|----------|----------|---------|--------|
| R1 | PII Masking | Security | Critical | Mensagem contém PII | Extrair → Persistir criptografado → Mascarar → Validar saída |
| R2 | Intent Detection | Business | High | Início da conversa | Classificar via LLM → Pedir confirmação se baixa confiança |
| R3 | Lead Qualification | Business | High | Lead fornece informações | Calcular score ponderado → Explicar em linguagem natural |
| R4 | RAG Trigger | Business | Medium | Score ≥ threshold OU lead pede | Buscar top-k imóveis via PropertiesRAG |
| R5 | Handoff Trigger | Business | High | Score ≥ threshold OU lead solicita | LeadRouter → Handoff → Desmascarar PII → Enviar ao corretor |
| R6 | Consent Recording | Legal | Critical | Primeira mensagem | Apresentar consentimento → Registrar → Encerrar se recusado |
| R7 | Guardrails | Security | High | Mensagem viola guardrails | Bloquear → Responder fallback → Log violação |
| R8 | Session Management | Technical | Medium | Mensagem do lead | Criar/Recuperar sessão → Expirar após 90 dias |

---

## Rule Details

### R1: PII Masking

**Purpose**: Proteger dados sensíveis conforme LGPD (nome, e-mail, telefone, CNPJ).

**Algorithm**:
1. Extrair PII via regex patterns:
   - Nome: `[A-Z][a-z]+ [A-Z][a-z]+`
   - E-mail: `[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}`
   - Telefone: `\+55\s?\d{2}\s?\d{4,5}-?\d{4}`
   - CNPJ: `\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}`
2. Persistir PII real no DynamoDB criptografado (KMS)
3. Substituir PII por placeholders: `[NOME]`, `[EMAIL]`, `[TELEFONE]`, `[CNPJ]`
4. Enviar texto mascarado ao LLM
5. Validar saída do LLM contra vazamento de PII (regex)
6. Se PII vazado, bloquear resposta e log erro

**Exception Handling**:
- Bloquear resposta se PII vazado
- Log erro para auditoria
- Responder com mensagem de fallback

---

### R2: Intent Detection

**Purpose**: Classificar intenção (compra/locação/investimento) com ≥ 85% de precisão.

**Algorithm**:
1. LLM classifica intenção com few-shot prompts
2. Se confiança ≥ threshold (0.85), aceitar classificação
3. Se confiança < threshold, pedir confirmação: "Você está buscando compra, locação ou investimento?"
4. Gravar intenção no Lead

**Exception Handling**:
- Perguntar confirmação se baixa confiança
- Não prosseguir até intenção confirmada

---

### R3: Lead Qualification

**Purpose**: Calcular score de prontidão com explainability.

**Algorithm**:
1. Coletar informações: metragem, região, orçamento, prazo, nº pessoas, decisor
2. Calcular score ponderado:
   - Prontidão (40%): informações completas, decisor definido
   - Urgência (30%): prazo curto (< 3 meses), orçamento definido
   - Ticket médio (30%): budget alto (> R$ 50k/mês)
3. Explicar score em linguagem natural: "Seu score é 85 porque você forneceu todas as informações e o prazo é curto"
4. Gravar score e fatores no Lead

**Exception Handling**:
- Se informações incompletas, perguntar faltantes
- Não qualificar se score < threshold (definido como 70)

---

### R4: RAG Trigger

**Purpose**: Buscar imóveis via RAG quando score ≥ threshold ou lead pede.

**Algorithm**:
1. Se score ≥ threshold OU lead pergunta por imóveis:
   - SDR Agent chama PropertiesRAG como tool
   - Buscar top-k imóveis compatíveis (k=3)
   - Filtros: metragem, região, orçamento
2. Nunca inventar imóveis que não estão na base (constraint via prompting)
3. Apresentar até 3 opções ao lead

**Exception Handling**:
- Se nenhum imóvel encontrado, explicar motivo: "Não encontramos imóveis com esses filtros"
- Sugerir ajustar filtros

---

### R5: Handoff Trigger

**Purpose**: Gerar handoff quando lead qualificado.

**Algorithm**:
1. Se score ≥ threshold OU lead solicita:
   - LeadRouter aplica regras de distribuição:
     - Até 500 m² → rodízio dos consultores
     - Acima → diretor/especialista
   - Handoff gera resumo Markdown: gap, score, intenção, urgência, próximos passos
   - Desmascarar PII apenas no destino (uso criptografado)
   - Enviar resumo ao corretor via Telegram/e-mail

**Exception Handling**:
- Se nenhum corretor disponível, notificar gestor
- Se handoff falhar, log erro e retry

---

### R6: Consent Recording

**Purpose**: Registrar consentimento LGPD na primeira mensagem.

**Algorithm**:
1. Lead envia `/start` ou primeira mensagem
2. Apresentar consentimento contextualizado:
   - Finalidade: atendimento SDR imobiliário
   - Dados coletados: nome, e-mail, telefone, CNPJ
   - Retenção: 90 dias
   - Compartilhamento: com corretor/CRM
   - Como recusar/revogar: responder "não"
3. Registrar consentimento na Conversation (consent_recorded=true)
4. Se recusado:
   - Não coletar dados
   - Encerrar educadamente: "Entendido. Se mudar de ideia, envie /start novamente"

**Exception Handling**:
- Sem exceção — consentimento é obrigatório

---

### R7: Guardrails

**Purpose**: Aplicar guardrails de tópicos negados e validação de entrada.

**Algorithm**:
1. Validar entrada contra prompt injection
2. Validar entrada contra tópicos negados (política, concorrentes, conteúdo ofensivo)
3. Se violação:
   - Bloquear conteúdo
   - Responder com mensagem de fallback: "Não posso ajudar com isso"
   - Log violação para auditoria

**Exception Handling**:
- Sem exceção — guardrails são obrigatórios

---

### R8: Session Management

**Purpose**: Gerenciar sessão com TTL de 90 dias.

**Algorithm**:
1. Lead envia mensagem
2. Se sessão não existe:
   - Criar nova sessão (session_id = UUID)
   - Criar novo Lead (lead_id = UUID)
   - Gravar no DynamoDB
3. Se sessão existe e não expirada:
   - Recuperar sessão do DynamoDB
   - Atualizar messages e context
4. Se sessão expirada (TTL 90 dias):
   - Recriar sessão
   - Explicar contexto perdido: "Vamos começar de novo, pois nossa conversa anterior expirou"

**Exception Handling**:
- Se sessão expirada, recriar e explicar contexto perdido