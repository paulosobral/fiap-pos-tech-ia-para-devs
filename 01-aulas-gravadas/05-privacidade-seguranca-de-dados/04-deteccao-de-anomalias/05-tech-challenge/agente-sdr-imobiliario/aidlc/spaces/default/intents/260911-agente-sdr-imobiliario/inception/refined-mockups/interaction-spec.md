# Interaction Specification — Agente SDR Imobiliário B2B

> Estágio Refined Mockups & UX Design (Inception). Fonte: mockups, user stories, refined-mockups-questions (Q2: padrões de interação).

---

## 1. Interactions — Dashboard Visão Geral (M1)

### 1.1 Refresh Manual (Q1-B)

**Trigger**: Botão [🔄 Refresh]

**Behavior**:
- Dispara requisição `GET /api/kpis` com filtros selecionados
- Exibe skeleton durante loading
- Atualiza KPIs, esteira Kanban, timeline e alertas
- Exibe erro toast se falhar

**Error Handling**:
- "Erro ao carregar dados. Tente novamente."
- Botão retry dispara nova requisição

---

### 1.2 Filtros Globais (Q1-B)

**Trigger**: Select dropdowns (Período, Canal, Status)

**Behavior**:
- Filtros não atualizam automaticamente (requer refresh manual)
- Valores: Período (Últimas 24h, Última semana, Último mês), Canal (Todos, Telegram, Email, Portal), Status (Todos, Novo, Qualificado, Em negociação, Anomalia)
- Filtros persistem em estado local (session storage)

---

### 1.3 Esteira Kanban

**Trigger**: Botão [Ver →] em cada coluna

**Behavior**:
- Abre modal com lista de leads daquele status
- Lista paginada (20 leads por página)
- Lead clicável abre detalhe do lead (M2)

---

### 1.4 Alertas de Anomalia

**Trigger**: Botão [Investigar →] em alerta

**Behavior**:
- Abre modal de anomalia detectada (M3)
- Permite ação do corretor (reclassificar ou manter suspeito)

**Trigger**: Botão [x] em alerta

**Behavior**:
- Marca alerta como investigado
- Remove da lista
- Atualiza contador de alertas não investigados

---

## 2. Interactions — Dashboard Detalhe do Lead (M2)

### 2.1 Navegação

**Trigger**: Botão [← Voltar]

**Behavior**:
- Retorna para visão geral (M1)
- Mantém filtros selecionados

---

### 2.2 Atualizar Status (Q2-A)

**Trigger**: Botão [Atualizar status]

**Behavior**:
- Abre modal de atualização de status
- Pre-select status atual
- Exibe motivação opcional

**Submit**:
- Dispara `POST /api/leads/{id}/status`
- Atualiza timeline com novo status
- Exibe toast "Status atualizado com sucesso"

**Cancel**:
- Fecha modal sem persistir

---

### 2.3 Agendar Visita (Q2-A)

**Trigger**: Botão [Agendar visita]

**Behavior**:
- Abre modal de agendamento
- Pre-popula imóveis recomendados do lead

**Submit**:
- Valida data/horário disponível no calendário simulado
- Dispara `POST /api/schedule`
- Gera convite ICS
- Notifica corretor via canal interno
- Exibe toast "Visita agendada com sucesso"
- Atualiza timeline com agendamento

**Cancel**:
- Fecha modal sem persistir

**View Calendar**:
- Abre modal com calendário visual (datas disponíveis marcadas)

---

### 2.4 Ver Imóveis Recomendados (Q2-A)

**Trigger**: Botão [Ver imóveis recomendados]

**Behavior**:
- Abre modal com lista de imóveis recomendados
- Exibe top-3 imóveis com detalhes (área, condomínio, localização, preço, disponibilidade)
- Botão [Ver mais opções] expande lista (se disponível)

---

### 2.5 Ver Resumo Handoff (Q2-A)

**Trigger**: Botão [Ver resumo handoff]

**Behavior**:
- Abre modal com resumo completo do handoff
- Exibe gap, score, urgência, próximos passos
- Botão [Ver conversa completa] abre modal com histórico de mensagens Telegram

---

### 2.6 Investigar Anomalia (Q2-A)

**Trigger**: Banner "⚠ Anomalia detectada" com botão [Investigar →]

**Behavior**:
- Abre modal de anomalia detectada (M3)
- Permite ação do corretor

---

## 3. Interactions — Modal Anomalia Detectada (M3)

### 3.1 Reclassificar como Lead

**Trigger**: Radio button "Reclassificar como lead (falso positivo)"

**Behavior**:
- Exibe campo de motivação obrigatório
- Remove flag de anomalia do lead
- Desbloqueia agendamento
- Registra timeline "Anomalia reclassificada como falso positivo"

**Submit**:
- Dispara `POST /api/leads/{id}/anomaly/reclassify`
- Fecha modal
- Exibe toast "Lead reclassificado com sucesso"

---

### 3.2 Manter como Suspeito

**Trigger**: Radio button "Manter como suspeito"

**Behavior**:
- Exibe campo de razão obrigatório
- Mantém flag de anomalia
- Bloqueia agendamento
- Registra timeline "Anomalia confirmada como suspeito"

**Submit**:
- Dispara `POST /api/leads/{id}/anomaly/confirm`
- Fecha modal
- Exibe toast "Anomalia confirmada"

---

## 4. Interactions — Integração HubSpot (M5)

### 4.1 Sincronizar Agora

**Trigger**: Botão [Sincronizar agora]

**Behavior**:
- Abre modal de autenticação MCP HubSpot
- Executa OAuth/PKCE (redirect local)
- Importa leads qualificados para HubSpot
- Atualiza contador de leads importados
- Exibe toast "Sincronização concluída"

**Error Handling**:
- "Erro na sincronização. Tente novamente."
- Botão retry

---

## 5. Telegram Interactions (Conversacional)

### 5.1 Iniciar Conversa (US1.1)

**Trigger**: Lead envia `/start` ou primeira mensagem

**Behavior**:
- Agente responde em < 10s com saudação humanizada
- Sistema cria sessão no DynamoDB com ID único
- Lead recebe consentimento LGPD contextualizado

**Lead recusa consentimento**:
- Sistema encerra educadamente
- Não coleta dados
- Registra recusa para auditoria

---

### 5.2 Texto Livre (US1.2)

**Trigger**: Lead envia mensagem de texto livre

**Behavior**:
- Agente processa conforme intents esperados (saudação, busca, agendamento, dúvida)
- Sistema mantém contexto conversacional entre mensagens
- Agente oferece botões inline como atalhos (mas não exige)

**Mensagem vazia**:
- Resposta de fallback: "Por favor, digite sua mensagem."

**Idioma não suportado**:
- Resposta de fallback: "No momento, aceito apenas português."

**Texto muito longo**:
- Resposta de fallback: "Sua mensagem é muito longa. Por favor, resuma."

---

### 5.3 Mensagem de Voz (US1.3)

**Trigger**: Lead envia voice message do Telegram

**Behavior**:
- Sistema recebe voice message via webhook
- Sistema baixa arquivo e converte para WAV (ffmpeg)
- Sistema transcreve áudio para texto (faster-whisper PT-BR)
- Texto transcrito entra no fluxo conversacional como mensagem digitada

**Transcrição > 15s**:
- Exibe indicador "Transcrevendo..."
- Aguarda completar antes de processar

---

## 6. Keyboard Navigation (WCAG 2.1 AA)

### 6.1 Tab Order

**Visão geral (M1)**:
1. Filtros (Período → Canal → Status)
2. Botão Refresh
3. Colunas da esteira Kanban (Novo → Qualificado → Em negociação → Anomalia)
4. Botões [Ver →] em cada coluna
5. Botões [Investigar →] em alertas

**Detalhe do lead (M2)**:
1. Botão Voltar
2. Select Status
3. Botões de ação (Ver imóveis → Ver resumo → Atualizar status → Agendar visita)
4. Botão Investigar anomalia (se presente)

**Modals**:
1. Botão fechar (✕)
2. Inputs (tab index order)
3. Botões de ação (Cancelar → Salvar/Agendar/Sincronizar)

---

### 6.2 Escape Key

**Behavior**:
- Fecha modal ativo sem persistir
- Retira foco para elemento anterior

---

### 6.3 Enter Key

**Behavior**:
- Em inputs: Submete formulário se botão principal disponível
- Em botões: Ativa botão focado

---

## 7. Error States

### 7.1 API Error

**Behavior**:
- Exibe toast "Erro ao carregar dados. Tente novamente."
- Botão retry disponível
- Logs erro no CloudWatch

---

### 7.2 Validation Error

**Behavior**:
- Exibe erro inline no campo (ex: "Data inválida")
- Botão salvar desabilitado até validação passar

---

### 7.3 Network Error

**Behavior**:
- Exibe toast "Erro de conexão. Verifique sua internet."
- Retry automático após 5s (max 3 tentativas)

---

## 8. Loading States

### 8.1 Refresh Manual

**Behavior**:
- Exibe skeleton cards durante loading
- Timeout: 30s
- Fallback: "Timeout ao carregar dados. Tente novamente."

---

### 8.2 Modal Loading

**Behavior**:
- Exibe spinner no botão de ação (ex: "Salvando...")
- Desabilita botão durante loading
- Timeout: 10s
- Fallback: "Timeout ao salvar. Tente novamente."

---

## 9. Toast Notifications

### 9.1 Success Toasts

- "Status atualizado com sucesso"
- "Visita agendada com sucesso"
- "Lead reclassificado com sucesso"
- "Anomalia confirmada"
- "Sincronização concluída"

**Style**: Green background, 3s auto-dismiss

---

### 9.2 Error Toasts

- "Erro ao carregar dados. Tente novamente."
- "Erro ao salvar. Tente novamente."
- "Erro na sincronização. Tente novamente."
- "Erro de conexão. Verifique sua internet."

**Style**: Red background, 5s auto-dismiss

---

### 9.3 Warning Toasts

- "Timeout ao carregar dados. Tente novamente."
- "Timeout ao salvar. Tente novamente."

**Style**: Yellow background, 5s auto-dismiss