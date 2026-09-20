# Refined Mockups Questions — Agente SDR Imobiliário B2B

## Q1: Representação das User Stories no Dashboard

Com base nas user stories must-have (US1.1-US7.1, US9.1, US11.1), como cada fluxo deve ser representado visualmente no dashboard Streamlit?

- **US1.1-US1.3 (Atendimento)**: Dashboard precisa mostrar sessões ativas em tempo real? Ou apenas histórico?
- **US2.1-US2.2 (Intenção e Qualificação)**: Como exibir score e intenção detectada na timeline do lead?
- **US3.1-US3.2 (Score e Handoff)**: Quando handoff é proposto, qual indicador visual no dashboard?
- **US4.1-US4.2 (RAG)**: Dashboard precisa mostrar quais imóveis foram recomendados ao lead?
- **US5.1 (Agendamento)**: Como visualizar compromissos agendados na timeline?
- **US6.1 (Handoff)**: Resumo do handoff deve ser visível para o corretor inline ou via modal?
- **US7.1 (Dashboard)**: KPIs e gráficos devem ser atualizados em tempo real ou com refresh manual?
- **US9.1 (Anomalias)**: Alertas de anomalia devem ser banners fixos ou uma seção dedicada?
- **US11.1 (CRM)**: Integração HubSpot deve ter botão dedicado ou integrado no fluxo de handoff?

**Opções**:
- A) Dashboard em tempo real com todas as informações inline
- B) Dashboard histórico com refresh manual, detalhes em modais
- C) Dashboard em tempo real para KPIs, histórico para leads detalhados
- D) Outro (especifique)

[Answer]: B

---

## Q2: Padrões de Interação

Quais padrões de interação são adequados para o dashboard Streamlit?

- **Inline actions**: Botões de ação (agendar, atualizar status, importar CRM) diretamente na linha do lead?
- **Modals/Wizards**: Qualificação de lead e agendamento em modals separados?
- **Progressive disclosure**: Informações detalhadas do lead expandidas sob demanda?
- **Wizards**: Formulários complexos (como atualização de status) em wizards passo-a-passo?

**Opções**:
- A) Ações inline, modals para formulários complexos
- B) Tudo inline, sem modals
- C) Modals para todas as ações
- D) Outro (especifique)

[Answer]: A

---

## Q3: Estados de Tela

Quais estados cada tela do dashboard deve suportar?

- **W1 (Visão geral)**: Loading, Empty (sem leads), Error (API falhou), Success
- **W2 (Detalhe do lead)**: Loading, Empty (lead não encontrado), Error, Success, Anomalia detectada
- **Telegram**: N/A (app de terceiros)
- **MCP Inspector**: N/A (ferramenta de demo)

**Opções**:
- A) Todos os estados implementados com skeletons e mensagens claras
- B) Apenas Loading e Success (outros estados simplificados)
- C) Apenas Loading, Empty e Success (error tratado globalmente)
- D) Outro (especifique)

[Answer]: A

---

## Q4: Design System / Component Library

O dashboard deve seguir qual design system?

- **Streamlit default**: Padrão limpo/neutro do Streamlit (como definido em wireframes)
- **Custom theme**: Tema customizado para alinhar com branding W Levitt
- **Component library**: Biblioteca de componentes específica (ex: Streamlit Elements)

**Opções**:
- A) Streamlit default (como definido em wireframes)
- B) Custom theme com branding W Levitt
- C) Component library específica
- D) Outro (especifique)

[Answer]: A

---

## Q5: Acessibilidade (WCAG)

Qual nível de conformidade WCAG é necessário para o dashboard?

- **WCAG 2.1 AA**: Nível padrão para aplicações web
- **WCAG 2.1 A**: Nível básico
- **WCAG 2.1 AAA**: Nível máximo (provavelmente overkill para POC)

**Opções**:
- A) WCAG 2.1 AA (recomendado)
- B) WCAG 2.1 A (básico)
- C) WCAG 2.1 AAA (máximo)
- D) Outro (especifique)

[Answer]: A

---

## Q6: Responsividade

Quais breakpoints são necessários?

- **Desktop only**: Dashboard focado em desktop (gestores e corretores usam desktop)
- **Mobile-friendly**: Responsivo para tablets e mobile
- **Desktop + Tablet**: Desktop e tablet, mobile simplificado

**Opções**:
- A) Desktop only (foco em gestores e corretores)
- B) Mobile-friendly (completo)
- C) Desktop + Tablet
- D) Outro (especifique)

[Answer]: A

---

## Q7: Developer Experience (API)

Para as APIs (conversation-router, RAG, qualificação, handoff, dashboard), como deve ser a experiência do desenvolvedor?

- **OpenAPI specification**: Documentação completa via Swagger/OpenAPI
- **Minimal docs**: Documentação básica em README
- **Auto-generated**: Documentação gerada automaticamente do código

**Opções**:
- A) OpenAPI specification completa
- B) Documentação básica em README
- C) Auto-generated do código
- D) Outro (especifique)

[Answer]: A