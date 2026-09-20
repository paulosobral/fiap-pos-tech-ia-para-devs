# Delivery Planning Questions — Agente SDR Imobiliário B2B

## Q1: Estratégia de Sequenciamento

O que devemos construir primeiro?

- **Partes mais arriscadas**: RAG, SecurityLayer (PII masking), Anomaly Detection
- **Partes mais valiosas**: Core Conversation (atendimento), Dashboard (visibilidade)
- **Thin end-to-end slice**: Walking skeleton que prove que a arquitetura funciona (US1.1-US1.2-US2.1-US2.2 básico)
- **Mix**: Walking skeleton primeiro, depois partes mais valiosas

**Opções**:
- A) Walking skeleton primeiro (thin end-to-end slice), depois partes mais valiosas
- B) Partes mais valiosas primeiro (Core Conversation → Dashboard)
- C) Partes mais arriscadas primeiro (SecurityLayer → RAG → Anomaly)
- D) Outro (especifique)

[Answer]: B

---

## Q2: Walking Skeleton

Qual deve ser o walking skeleton?

- **Conversação básica**: US1.1 (iniciar conversa) + US1.2 (texto livre) + US2.1 (detectar intenção) — prova que Telegram → Lambda → LLM funciona
- **Conversação com RAG**: Walking skeleton acima + US4.1 (buscar imóveis) — prova que RAG FAISS funciona
- **Conversão completa**: Todas as US1-US7 básicas — prova todo o fluxo end-to-end

**Opções**:
- A) Conversação básica (Telegram → Lambda → LLM)
- B) Conversação com RAG (Telegram → Lambda → LLM → FAISS)
- C) Conversação completa (todas as US1-US7 básicas)
- D) Outro (especifique)

[Answer]: A

---

## Q3: Tolerância a Risco

Qual é a tolerância a risco para sequenciamento?

- **Baixa**: Foco em walking skeleton primeiro, validar tudo antes de avançar
- **Média**: Walking skeleton + parts mais valiosas em paralelo
- **Alta**: Começar com parts mais arriscadas para validar cedo

**Opções**:
- A) Baixa (validar tudo antes de avançar)
- B) Média (walking skeleton + paralelo)
- C) Alta (validar cedo riscos)
- D) Outro (especifique)

[Answer]: A

---

## Q4: Alocação de Equipe

Como a equipe deve ser alocada para os Bolts?

- **1 pessoa sequencial**: Foco em um Bolt por vez (ideal para hackathon)
- **2 pessoas paralelas**: Dividir Bolts independentes em paralelo
- **3+ pessoas**: Cada Bolt dedicado, máximo paralelismo

**Opções**:
- A) 1 pessoa sequencial (ideal para hackathon)
- B) 2 pessoas paralelas
- C) 3+ pessoas dedicadas
- D) Outro (especifique)

[Answer]: A

---

## Q5: Dependências Externas

Como lidar com dependências externas (Telegram, OpenRouter, HubSpot)?

- **Mockar tudo**: Usar mocks para todas as dependências externas na POC
- **Integrar reais**: Integrar com Telegram e OpenRouter, mockar apenas HubSpot
- **Integrar tudo**: Integrar com todas as dependências reais

**Opções**:
- A) Mockar tudo (mais seguro, menos real)
- B) Integrar reais (Telegram + OpenRouter), mockar HubSpot (conforme PRD)
- C) Integrar tudo (mais real, mais arriscado)
- D) Outro (especifique)

[Answer]: A