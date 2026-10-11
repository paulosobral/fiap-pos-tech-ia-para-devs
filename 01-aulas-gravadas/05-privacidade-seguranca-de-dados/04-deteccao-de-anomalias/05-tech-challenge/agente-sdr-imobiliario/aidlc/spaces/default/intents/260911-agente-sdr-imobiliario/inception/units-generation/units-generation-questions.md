# Units Generation Questions — Agente SDR Imobiliário B2B

## Q1: Estratégia de Limites de Unidade

Como agrupar os componentes em unidades de trabalho (Bolts)?

- **Por serviço**: Cada Lambda/deployment é uma unidade (alinhado com PRD §7.1)
- **Por feature**: Cada funcionalidade principal é uma unidade (ex: conversação, RAG, agendamento)
- **Por domínio**: Cada domínio de negócio é uma unidade (ex: atendimento, CRM, dashboard)
- **Por deployment target**: Cada infraestrutura é uma unidade (AWS Lambda, SQS, EventBridge, etc.)

**Opções**:
- A) Por deployment target (alinhado com PRD §7.1 — cada Lambda é uma unidade)
- B) Por serviço (cada componente lógico é uma unidade)
- C) Por feature (cada funcionalidade principal é uma unidade)
- D) Outro (especifique)

[Answer]: A

---

## Q2: Granularidade das Unidades

Qual nível de granularidade para as unidades?

- **Coarse-grained**: Poucas unidades grandes (1-3 unidades, cada com múltiplos componentes)
- **Medium-grained**: Unidades de tamanho médio (4-8 unidades)
- **Fine-grained**: Muitas unidades pequenas (9+ unidades, cada com 1-2 componentes)

**Opções**:
- A) Coarse-grained (foco em walking skeleton rápido)
- B) Medium-grained (equilíbrio entre simplicidade e paralelismo)
- C) Fine-grained (máximo paralelismo, mais overhead de coordenação)
- D) Outro (especifique)

[Answer]: A

---

## Q3: Ordenação de Dependências

Como definir a ordem de dependências entre unidades?

- **Estrita topológica**: Unidades só podem começar quando todas as dependências estão completas
- **Paralelismo permitido**: Unidades independentes podem ser desenvolvidas em paralelo
- **Híbrido**: Estrita para críticas, paralelo para não-críticas

**Opções**:
- A) Estrita topológica (simples, sequencial)
- B) Paralelismo permitido (mais rápido, mais coordenação)
- C) Híbrido (equilíbrio)
- D) Outro (especifique)

[Answer]: A

---

## Q4: Pontos de Integração

Como as unidades se integram?

- **APIs HTTP**: Comunicação via HTTP/REST (síncrono)
- **Shared Data**: Acesso compartilhado a DynamoDB/S3
- **Events**: Comunicação via SQS/EventBridge (assíncrono)
- **Híbrido**: Mistura de APIs, shared data e events

**Opções**:
- A) APIs HTTP para tudo (síncrono)
- B) Shared Data para tudo (banco compartilhado)
- C) Events para tudo (assíncrono)
- D) Híbrido (alinhado com PRD §7.1 — síncrono para core, assíncrono para slow)

[Answer]: D

---

## Q5: Modelo de Deployment

Como as unidades são deployadas?

- **Monolithic deploy**: Tudo deployado junto (um comando só)
- **Independent deploy**: Cada unidade deployada independentemente
- **Hybrid**: Core monolithic, periféricos independentes

**Opções**:
- A) Monolithic deploy (simples, start.sh único)
- B) Independent deploy (mais flexível, mais complexo)
- C) Hybrid (core monolithic, periféricos independentes)
- D) Outro (especifique)

[Answer]: A

---

## Summary Confirmation

**Resumo consolidado das respostas**:
- **Q1 (Estratégia de limites)**: A - Por deployment target (alinhado com PRD §7.1 — cada Lambda é uma unidade)
- **Q2 (Granularidade)**: A - Coarse-grained (foco em walking skeleton rápido)
- **Q3 (Ordenação de dependências)**: A - Estrita topológica (simples, sequencial)
- **Q4 (Pontos de integração)**: D - Híbrido (alinhado com PRD §7.1 — síncrono para core, assíncrono para slow)
- **Q5 (Modelo de deployment)**: A - Monolithic deploy (simples, start.sh único)