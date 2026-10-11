# User Stories Plan Questions

## Sources

- [req] Requirements: `requirements.md` — 12 FRs, 9 NFRs, priorização must-have vs nice-to-have já definida
- [prd] PRD preliminar: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`
- [intent] Intent Capture: Personas definidas (Lead B2B, Investidor PF, Corretor Especialista, Gestor)

> Planejamento de user stories baseado nos requisitos consolidados e nas personas definidas no AI-DLC.

---

## Q1. Desenvolvimento de Personas

Quantas personas principais de usuário devem ser definidas?

- A. 3 personas: Lead B2B (principal), Investidor PF (secundário), Gestor W Levitt (monitoramento)
- B. 4 personas: Lead B2B, Investidor PF, Corretor Especialista, Gestor W Levitt
- C. 2 personas: Lead B2B e Gestor W Levitt (focar no principal)
- X. Other (especificar)

[Answer]: A

---

## Q2. Formato das Stories

As user stories devem seguir o formato padrão "Como [persona], quero [goal], para que [benefit]" com critérios de aceitação?

- A. Sim, formato padrão com critérios de aceitação testáveis
- B. Simplificar — apenas título e descrição sem critérios detalhados
- C. Mais detalhado — incluir personas adicionais e edge cases
- X. Other (especificar)

[Answer]: A

---

## Q3. Priorização MoSCoW

As stories devem usar priorização MoSCoW (Must Have / Should Have / Could Have / Won't Have)?

- A. Sim, alinhado com must-have vs nice-to-have já definido nos requisitos
- B. Não priorizar agora — decidir no Delivery Planning
- C. Priorizar por sequenciamento de dependências apenas
- X. Other (especificar)

[Answer]: A

---

## Q4. Abordagem de Decomposição

Como decompor as stories?

- A. Por feature/FR (cada FR principal vira um grupo de stories)
- B. Por persona (stories agrupadas por persona principal)
- C. Por workflow (stories seguindo o fluxo do lead: contato → qualificação → agendamento)
- D. Por epic (stories agrupadas em grandes épicos funcionais)
- X. Other (especificar)

[Answer]: A

---

## Q5. Granularidade das Stories

Qual o tamanho ideal das stories?

- A. Stories pequenas (1-3 dias cada) — melhor para track de progresso
- B. Stories médias (3-5 dias cada) — equilíbrio entre detalhe e gestão
- C. Stories grandes (5-7 dias cada) — focar em funcionalidades completas
- X. Other (especificar)

[Answer]: A

---

## Q6. Cobertura de Requisitos

As stories devem cobrir todos os FRs ou apenas os must-have?

- A. Todos os FRs (must-have + nice-to-have) com priorização apropriada
- B. Apenas must-have inicialmente, nice-to-have se sobrar tempo
- C. Focus em must-have + 1-2 nice-to-have high-value
- X. Other (especificar)

[Answer]: B

---

## Q7. Personas Adicionais

Devemos incluir personas internas (Corretor Especialista, SDR humano) além das externas?

- A. Sim, personas internas são importantes para handoff e operação
- B. Não, focar apenas em personas externas (leads e gestor)
- C. Incluir apenas Corretor Especialista (mais crítico que SDR)
- X. Other (especificar)

[Answer]: B

---

## Assumptions & Open Questions

- INVEST criteria será aplicado: Independent, Negotiable, Valuable, Estimable, Small, Testable
- Critérios de aceitação serão testáveis e mensuráveis
- Traceabilidade com FRs será mantida via traceability.json

## Review

Este arquivo é o registro do planejamento de user stories. As respostas guiarão a geração de personas.md, stories.md e traceability.json.

## Consolidated Summary Confirmation

Does this all look correct before I generate the user stories plan?

- Looks correct
- Request changes

[Answer]: Looks correct
