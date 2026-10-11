# Refined Mockups — Agente SDR Imobiliário B2B

> Estágio Refined Mockups & UX Design (Inception). Fonte: wireframes, user-flow, user stories, requirements, refined-mockups-questions (Q1–Q7).
> Decisões de design: Dashboard histórico com refresh manual (Q1-B), ações inline + modals (Q2-A), todos os estados (Q3-A), Streamlit default (Q4-A), WCAG 2.1 AA (Q5-A), desktop only (Q6-A), OpenAPI specification (Q7-A).

---

## M1. Dashboard — Visão Geral (KPIs Agregados)

**Refinamento sobre W1**: Dashboard histórico com refresh manual, todos os estados implementados.

```
+------------------------------------------------------------------+
|| [W Levitt · Agente SDR]  Filtros: [Período v] [Canal v] [Status v]|
||                       [🔄 Refresh]                              |
+------------------------------------------------------------------+
||  KPI: Leads    | KPI: Qualificados | KPI: Intenção    | KPI: 1ª   |
||  1.248         | 62%               | 87%              | resp. 4s  |
||  (±12% vs semana passada)                                   |
+------------------------------------------------------------------+
||  Esteira Kanban (status dos leads)                               |
||  +-----------+ +-----------+ +-----------+ +-----------+        |
||  | Novo      | | Qualif.   | | Em negoc. | | Anomalia  |        |
||  | 312       | | 180       | | 96        | | 7         |        |
||  | [Ver →]   | | [Ver →]   | | [Ver →]   | | [Ver →]   |        |
||  +-----------+ +-----------+ +-----------+ +-----------+        |
+------------------------------------------------------------------+
||  [Timeline do lead selecionado]   |  [Alertas de anomalia]       |
||  entrada → qualificação → rota →  |  ⚠ Lead #4821 suspeito     |
||  status → anomalia                |  ⚠ Pico de intenção falsa  |
||                                    |  [Investigar →]            |
+------------------------------------------------------------------+
```

**Estados implementados (Q3-A)**:
- **Loading**: Skeleton cards durante refresh manual
- **Empty**: "Nenhum lead no período selecionado" com ilustração
- **Error**: "Erro ao carregar dados. Tente novamente." com botão retry
- **Success**: Tela principal acima

**Acessibilidade (Q5-A - WCAG 2.1 AA)**:
- `h1` no título principal
- Landmarks `header`, `main`, `aside`
- Entrada por teclado em todos os filtros e botões
- Contraste mínimo 4.5:1 para texto
- Foco visível em todos os elementos interativos
- Texto alternativo para ícones

---

## M2. Dashboard — Detalhe do Lead (Timeline + Ação)

**Refinamento sobre W2**: Detalhes em modal (Q1-B), ações inline + modals (Q2-A).

```
+------------------------------------------------------------------+
|| ← Voltar   Lead #4821 · Empresa XYZ · laje corporativa 800m²     |
+------------------------------------------------------------------+
||  Status: [Qualificado v]   Intenção: Locação   Score: 82/100   |
+------------------------------------------------------------------+
||  Timeline:                                                       |
||  • 14:02 entrada (Telegram) ✅                                  |
||  • 14:03 qualificação (score 82) ✅                              |
||  • 14:04 rota roleta → corretor A (regra: score≥70) ✅          |
||  • 14:05 handoff enviado ao corretor ✅                         |
||  • 14:06 ⚠ anomalia detectada (job diário) [Investigar →]      |
+------------------------------------------------------------------+
||  [Ver imóveis recomendados]  [Ver resumo handoff]              |
||  [Atualizar status]  [Agendar visita]                           |
+------------------------------------------------------------------+
```

**Modal: Imóveis Recomendados (Q2-A)**

```
+---------------------------------------------+
| Imóveis Recomendados (3)             [✕]    |
+---------------------------------------------+
| Imóvel 1: Torre A - 800m²                 |
| • Área útil: 750m² | Área bruta: 850m²    |
| • Condomínio: R$ 15/m²                    |
| • Localização: Paulista, 1000             |
| • Preço: R$ 45.000/m² (locação)           |
| • Disponibilidade: Imediata               |
|                                             |
| Imóvel 2: Torre B - 750m²                 |
| • Área útil: 700m² | Área bruta: 800m²    |
| • Condomínio: R$ 12/m²                    |
| • Localização: Faria Lima, 2000           |
| • Preço: R$ 38.000/m² (locação)           |
| • Disponibilidade: Em 30 dias             |
|                                             |
| Imóvel 3: Torre C - 900m²                 |
| • Área útil: 850m² | Área bruta: 950m²    |
| • Condomínio: R$ 18/m²                    |
| • Localização: Berrini, 1500              |
| • Preço: R$ 52.000/m² (locação)           |
| • Disponibilidade: Imediata               |
|                                             |
| [Ver mais opções]                          |
+---------------------------------------------+
```

**Modal: Resumo Handoff (Q2-A)**

```
+---------------------------------------------+
| Resumo Handoff ao Corretor          [✕]    |
+---------------------------------------------+
| Lead: Empresa XYZ                          |
| Intenção: Locação corporativa              |
| Score: 82/100 (quente)                     |
| Urgência: Alta                             |
|                                             |
| Gap:                                        |
| • Metragem: 800m²                          |
| • Região: Paulista / Faria Lima            |
| • Orçamento: R$ 35.000/m² mensal           |
| • Prazo: Imediato                          |
| • Nº pessoas: 50                          |
| • Decisor: Diretor de Operações            |
|                                             |
| Próximos passos:                            |
| • Agendar visita aos imóveis recomendados   |
| • Proposta comercial                       |
|                                             |
| [Ver conversa completa]  [Fechar]          |
+---------------------------------------------+
```

**Modal: Atualizar Status (Q2-A)**

```
+---------------------------------------------+
| Atualizar Status do Lead             [✕]    |
+---------------------------------------------+
| Lead #4821 · Empresa XYZ                     |
|                                             |
| Status atual: Qualificado                   |
|                                             |
| Novo status: [Qualificado v]                |
|              Em negociação                  |
|              Fechado                        |
|              Perdido                        |
|                                             |
| Motivo: [____________________________]      |
|                                             |
| [Cancelar]  [Salvar]                        |
+---------------------------------------------+
```

**Modal: Agendar Visita (Q2-A)**

```
+---------------------------------------------+
| Agendar Visita                      [✕]    |
+---------------------------------------------+
| Lead #4821 · Empresa XYZ                     |
|                                             |
| Imóvel: [Torre A - 800m² v]                 |
|                                             |
| Data: [20/09/2026]                           |
| Horário: [14:00 v]                          |
|                                             |
| Notas: [____________________________]      |
|                                             |
| [Cancelar]  [Agendar]  [Ver calendário]    |
+---------------------------------------------+
```

**Estados implementados (Q3-A)**:
- **Loading**: Skeleton durante carregamento do lead
- **Empty**: "Lead não encontrado" com botão voltar
- **Error**: "Erro ao carregar lead. Tente novamente." com retry
- **Success**: Tela principal acima
- **Anomalia detectada**: Banner fixo "⚠ Anomalia detectada" com botão investigar

**Acessibilidade (Q5-A - WCAG 2.1 AA)**:
- `h2` nos blocos de conteúdo
- Landmarks `main`, `section`
- Foco visível em todos os botões de ação
- Modals com trap de foco
- Labels associados a todos os inputs
- Mensagens de erro com `role="alert"`

---

## M3. Modal: Anomalia Detectada (Ação do Corretor)

**Novo fluxo** (baseado em wireframes W2 + decisão Q2-A):

```
+---------------------------------------------+
| ⚠ Anomalia Detectada                [✕]    |
+---------------------------------------------+
| Lead #4821 · Empresa XYZ                     |
|                                             |
| Tipo de anomalia: Pico de intenção falsa    |
| Confiança: 87%                              |
|                                             |
| Razão:                                      |
| • Volume de mensagens atípico (12x média)   |
| • Horários de atividade noturnos            |
| • Comprimento de mensagens uniforme         |
|                                             |
| Ação necessária:                            |
|                                             |
| ○ Reclassificar como lead (falso positivo)   |
|   Motivo: [________________________]        |
|                                             |
| ● Manter como suspeito                      |
|   Razão: [________________________]          |
|                                             |
| [Cancelar]  [Salvar decisão]                 |
+---------------------------------------------+
```

**Ações disponíveis (Q2-A)**:
- Reclassificar como lead: Remove flag de anomalia, desbloqueia agendamento
- Manter como suspeito: Mantém flag, bloqueia agendamento, registra timeline

---

## M4. Dashboard — Alertas de Anomalia (Seção Dedicada)

**Refinamento sobre W1**: Seção dedicada para alertas (Q1-B, Q9.1).

```
+------------------------------------------------------------------+
|| [Alertas de Anomalia]                              [Ver todos →]|
+------------------------------------------------------------------+
||  ⚠ Lead #4821 · Empresa XYZ · Pico de intenção falsa  [x]     |
||     Detectado: 20/09/2026 14:06 | Confiança: 87%                |
||     [Investigar →]                                            |
||                                                                 |
||  ⚠ Lead #5234 · Startup ABC · Volume atípico       [x]        |
||     Detectado: 20/09/2026 13:45 | Confiança: 92%                |
||     [Investigar →]                                            |
||                                                                 |
||  ⚠ Lead #6102 - Empresa DEF · Horários atípicos     [x]        |
||     Detectado: 20/09/2026 12:30 | Confiança: 78%                |
||     [Investigar →]                                            |
+------------------------------------------------------------------+
```

**Comportamento**:
- Alertas mais recentes no topo
- Marcar como "investigado" remove da lista
- Contador de alertas não investigados no badge
- Auto-refresh manual via botão 🔄

---

## M5. Dashboard — Integração HubSpot (Q7-A)

**Botão dedicado** (Q1-B, Q11.1):

```
+------------------------------------------------------------------+
|| [🔗 Importar via MCP HubSpot]                                    |
+------------------------------------------------------------------+
||  Status: Disponível                                             |
||  Última importação: 20/09/2026 10:30                           |
||  Leads importados: 1.248                                        |
||                                                                 |
||  [Sincronizar agora]                                            |
+------------------------------------------------------------------+
```

**Modal: Sincronização HubSpot (Q2-A)**

```
+---------------------------------------------+
| Sincronização HubSpot               [✕]    |
+---------------------------------------------+
| Método: MCP HubSpot (demo única ao vivo)     |
|                                             |
| Auth: OAuth/PKCE (redirect local)            |
|                                             |
| [Conectar HubSpot]                           |
|                                             |
| Nota: Esta é uma demonstração única ao vivo. |
| Em produção, usar Private App REST.         |
+---------------------------------------------+
```

---

## M6. Dashboard — Filtros (Q1-B)

**Filtros globais** na visão geral:

```
Período: [Últimas 24h v]
Canal: [Todos v] (Telegram, Email, Portal)
Status: [Todos v] (Novo, Qualificado, Em negociação, Anomalia)
[🔄 Refresh]
```

**Filtros aplicam a**:
- KPIs (leads, qualificados, intenção, tempo de resposta)
- Esteira Kanban
- Timeline
- Alertas de anomalia

---

## Design System Mapping (Q4-A)

**Streamlit Default** (como definido em wireframes):

| Componente | Streamlit Component | Estilo Default |
|------------|---------------------|----------------|
| Título | `st.title()` | Streamlit blue theme |
| Subtítulo | `st.header()` | Streamlit default |
| Texto | `st.write()` | Streamlit default |
| Input de texto | `st.text_input()` | Streamlit default |
| Select | `st.selectbox()` | Streamlit default |
| Botão | `st.button()` | Streamlit default |
| Tabela | `st.dataframe()` | Streamlit default |
| Chart | `st.line_chart()`, `st.bar_chart()` | Streamlit default |
| Modal | `st.dialog()` (Streamlit 1.30+) | Streamlit default |
| Alerta | `st.error()`, `st.warning()`, `st.success()` | Streamlit default |
| Skeleton | Custom loading spinner | `st.spinner()` |

**Cores** (Streamlit default):
- Primary: `#4F8EF7` (Streamlit blue)
- Background: `#FFFFFF`
- Text: `#262730`
- Success: `#00C851`
- Warning: `#FFBB33`
- Error: `#FF4444`

**Tipografia** (Streamlit default):
- Fonte: Sans-serif (system font stack)
- H1: 36px, bold
- H2: 28px, bold
- H3: 22px, bold
- Body: 16px, regular

---

## Responsividade (Q6-A)

**Desktop Only** (foco em gestores e corretores):

- **Desktop** (1024px+): Layout completo, 2 colunas
- **Tablet** (768px-1023px): Layout adaptado, 1 coluna
- **Mobile** (<768px): Não suportado (UX otimizada para desktop)

**Breakpoints**:
- Desktop: 1024px+ (foco principal)
- Tablet: 768px-1023px (suporte secundário)
- Mobile: <768px (não implementado)

---

## Nota

- **Telegram** (W3): App existente, sem UI a desenvolver — apenas comportamento conversacional do bot
- **MCP Inspector** (W4): Demo única ao vivo, ferramenta de demonstração, não UI do produto
- **Dashboard** (M1-M6): Única UI a desenvolver (Streamlit, desktop only)