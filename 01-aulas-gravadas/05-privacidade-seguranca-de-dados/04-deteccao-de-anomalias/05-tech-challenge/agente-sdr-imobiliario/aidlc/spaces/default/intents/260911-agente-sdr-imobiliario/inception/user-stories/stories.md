# User Stories — Agente SDR Imobiliário B2B

## Grupo 1: Atendimento Conversacional (FR1)

### US1.1 — Iniciar Conversa
**Como** Lead B2B, **quero** iniciar uma conversa com o agente SDR via Telegram, **para que** eu possa obter informações sobre imóveis corporativos sem esperar horário comercial.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: Nenhuma

**Acceptance Criteria**:
- AC1.1.1: Lead envia `/start` ou primeira mensagem no Telegram
- AC1.1.2: Agente responde em < 10s com saudação humanizada
- AC1.1.3: Sistema cria sessão no DynamoDB com ID único
- AC1.1.4: Lead recebe consentimento LGPD contextualizado na primeira mensagem

**INVEST Notes**: Independent (não depende de outras stories), Negotiable (detalhes da saudação podem ajustar), Valuable (first touchpoint), Estimable (clear boundary), Small (focused), Testable (clear acceptance criteria)

---

### US1.2 — Conversar com Texto Livre
**Como** Lead B2B, **quero** digitar mensagens de texto livre para o agente, **para que** eu possa expressar minhas necessidades de forma natural sem seguir um menu rígido.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US1.1

**Acceptance Criteria**:
- AC1.2.1: Agente aceita texto livre e processa naturalmente
- AC1.2.2: Sistema mantém contexto conversacional entre mensagens
- AC1.2.3: Agente oferece botões inline como atalhos (mas não exige)
- AC1.2.4: Contexto é persistido no DynamoDB com TTL 90 dias

**INVEST Notes**: Independent (after US1.1), Negotiable (inline buttons optional), Valuable (natural UX), Estimable (clear), Small (focused), Testable (verifiable)

---

### US1.3 — Enviar Mensagem de Voz
**Como** Lead B2B, **quero** enviar mensagens de voz para o agente, **para que** eu possa descrever minhas necessidades mais rapidamente do que digitando.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US1.1

**Acceptance Criteria**:
- AC1.3.1: Sistema recebe voice message do Telegram via webhook
- AC1.3.2: Sistema baixa arquivo e converte para WAV (ffmpeg)
- AC1.3.3: Sistema transcreve áudio para texto (faster-whisper PT-BR)
- AC1.3.4: Texto transcrito entra no fluxo conversacional como mensagem digitada
- AC1.3.5: Transcrição acontece em < 15s para áudios < 30s

**INVEST Notes**: Independent (after US1.1), Negotiable (timeout acceptable), Valuable (convenience), Estimable (clear technical path), Small (focused), Testable (verifiable)

---

## Grupo 2: Identificação de Intenção (FR2)

### US2.1 — Detectar Intenção
**Como** Agente SDR, **quero** identificar a intenção do lead (compra/locação/investimento) no início da conversa, **para que** eu possa adaptar o fluxo de qualificação apropriado.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US1.2

**Acceptance Criteria**:
- AC2.1.1: Sistema classifica intenção com >= 85% de precisão
- AC2.1.2: Classificação acontece nas primeiras 3 trocas de mensagem
- AC2.1.3: Lead B2B é diferenciado de Investidor PF automaticamente
- AC2.1.4: Sistema registra intenção detectada no DynamoDB

**INVEST Notes**: Independent (after US1.2), Negotiable (3 turns acceptable), Valuable (routing accuracy), Estimable (clear metric), Small (focused), Testable (verifiable)

---

### US2.2 — Coletar Informações Básicas
**Como** Agente SDR, **quero** coletar informações básicas do lead (metragem, região, orçamento, prazo, nº pessoas, decisor), **para que** eu possa qualificar o lead adequadamente.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US2.1

**Acceptance Criteria**:
- AC2.2.1: Sistema aplica questionário adaptativo baseado em intenção detectada
- AC2.2.2: Para compra/locação: coleta metragem, região, orçamento, prazo, nº pessoas, decisor
- AC2.2.3: Para investimento: coleta ticket médio, expectativa de retorno (%), período
- AC2.2.4: Informações são estruturadas e persistidas no DynamoDB
- AC2.2.5: Questionário não repete perguntas já respondidas

**INVEST Notes**: Independent (after US2.1), Negotiable (question order flexible), Valuable (qualification data), Estimable (clear fields), Small (focused), Testable (verifiable)

---

## Grupo 3: Qualificação de Leads (FR3)

### US3.1 — Calcular Score de Prontidão
**Como** Agente SDR, **quero** calcular score de prontidão e urgência do lead, **para que** eu possa priorizar leads mais quentes para os corretores.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US2.2

**Acceptance Criteria**:
- AC3.1.1: Sistema calcula score baseado em respostas (orçamento, prazo, urgência declarada)
- AC3.1.2: Score vai de 0-100, onde >= 70 indica lead quente
- AC3.1.3: Sistema identifica urgência (alta/média/baixa) automaticamente
- AC3.1.4: Score e urgência são registrados no DynamoDB

**INVEST Notes**: Independent (after US2.2), Negotiable (scoring algorithm flexible), Valuable (prioritization), Estimable (clear logic), Small (focused), Testable (verifiable)

---

### US3.2 — Qualificar para Handoff
**Como** Agente SDR, **quero** identificar quando o lead está suficientemente qualificado para handoff ao corretor, **para que** eu possa transferir leads prontos sem perder tempo de corretores.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.1

**Acceptance Criteria**:
- AC3.2.1: Sistema propõe handoff quando score >= 70 OU lead solicita
- AC3.2.2: Sistema coleta informações restantes necessárias antes do handoff
- AC3.2.3: Lead recebe confirmação antes do handoff
- AC3.2.4: Sistema bloqueia handoff de leads com anomalias detectadas

**INVEST Notes**: Independent (after US3.1), Negotiable (threshold adjustable), Valuable (efficiency), Estimable (clear logic), Small (focused), Testable (verifiable)

---

## Grupo 4: RAG sobre Base de Imóveis (FR4)

### US4.1 — Buscar Imóveis Relevantes
**Como** Lead B2B, **quero** receber recomendações de imóveis compatíveis com meus critérios, **para que** eu possa visualizar opções relevantes sem ter que buscar manualmente.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US2.2

**Acceptance Criteria**:
- AC4.1.1: Sistema busca top-k imóveis na base sintética (100–200 ofertas)
- AC4.1.2: Busca usa filtros do lead (metragem, região, orçamento, tipo)
- AC4.1.3: Sistema nunca inventa imóveis que não estão na base (constraint via RAG)
- AC4.1.4: Índice FAISS é carregado em memória da Lambda
- AC4.1.5: Busca acontece em < 2s

**INVEST Notes**: Independent (after US2.2), Negotiable (k-value adjustable), Valuable (relevant recommendations), Estimable (clear performance), Small (focused), Testable (verifiable)

---

### US4.2 — Apresentar Opções
**Como** Lead B2B, **quero** ver até 3 opções de imóveis com detalhes relevantes, **para que** eu possa comparar e escolher a melhor opção.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US4.1

**Acceptance Criteria**:
- AC4.2.1: Sistema apresenta até 3 imóveis com área útil, área bruta, condomínio, localização
- AC4.2.2: Cada opção inclui preço e disponibilidade
- AC4.2.3: Lead pode escolher uma opção ou pedir mais opções
- AC4.2.4: Sistema adapta apresentação baseado em intenção (compra vs locação vs investimento)

**INVEST Notes**: Independent (after US4.1), Negotiable (3 options adjustable), Valuable (comparison capability), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 5: Agendamento de Reuniões (FR5)

### US5.1 — Agendar Visita
**Como** Lead B2B, **quero** agendar uma visita ao imóvel escolhido, **para que** eu possa conhecer o espaço pessoalmente antes de decidir.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US4.2

**Acceptance Criteria**:
- AC5.1.1: Sistema valida data/hora disponível no calendário simulado
- AC5.1.2: Sistema grava compromisso no calendário simulado
- AC5.1.3: Sistema emite convite ICS para o lead
- AC5.1.4: Sistema notifica corretor via canal interno (Telegram/e-mail)
- AC5.1.5: Lead recebe confirmação do agendamento

**INVEST Notes**: Independent (after US4.2), Negotiable (validation logic flexible), Valuable (convenience), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 6: Resumo Inteligente (FR6)

### US6.1 — Gerar Handoff
**Como** Corretor Especialista, **quero** receber um resumo inteligente do lead qualificado, **para que** eu possa ter contexto completo antes de iniciar o contato sem ter que ler toda a conversa.

**Priority**: Must Have  
**Persona**: Corretor Especialista (interna)  
**Depends**: US3.2

**Acceptance Criteria**:
- AC6.1.1: Sistema gera handoff em Markdown com gap, score, intenção, urgência, próximos passos
- AC6.1.2: Sistema envia mensagem interna + arquivo de resumo ao corretor
- AC6.1.3: Sistema desmascara PII apenas no handoff interno (uso criptografado)
- AC6.1.4: Handoff inclui todas as informações coletadas (metragem, região, orçamento, etc.)
- AC6.1.5: Handoff é gerado automaticamente quando lead atinge score >= 70

**INVEST Notes**: Independent (after US3.2), Negotiable (format flexible), Valuable (efficiency), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 7: Dashboard Mínimo (FR7)

### US7.1 — Monitorar KPIs
**Como** Gestor W Levitt, **quero** ver métricas de operação em tempo real, **para que** eu possa tomar decisões informadas sobre investimentos e operação.

**Priority**: Must Have  
**Persona**: Gestor W Levitt  
**Depends**: Nenhuma (paralelo)

**Acceptance Criteria**:
- AC7.1.1: Dashboard consome `GET /api/kpis` com métricas de negócio
- AC7.1.2: Dashboard exibe linha de métricas: leads hoje/semana, tempo 1ª resposta p90, taxa qualificação, agendamentos
- AC7.1.3: Dashboard exibe gráficos: volume de intenções, leads distribuídos pela roleta
- AC7.1.4: Dashboard exibe tabela de anomalias com alertas 24h
- AC7.1.5: Dashboard é acessível via login Cognito (JWT)

**INVEST Notes**: Independent (parallel), Negotiable (UI layout flexible), Valuable (visibility), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 8: Detecção de Anomalias (FR9)

### US9.1 — Detectar Comportamento Anômalo
**Como** Gestor W Levitt, **quero** receber alertas automáticos de comportamentos suspeitos, **para que** eu possa investigar e mitigar riscos de fraude ou bots.

**Priority**: Must Have  
**Persona**: Gestor W Levitt  
**Depends**: US7.1

**Acceptance Criteria**:
- AC9.1.1: Job diário extrai features por conversa (volume, comprimento, sentimento, horários atípicos)
- AC9.1.2: Sistema aplica Isolation Forest + PCA + Autoencoder
- AC9.1.3: Sistema emite alerta no dashboard quando anomalia é detectada
- AC9.1.4: Sistema restringe agendamento para leads suspeitos
- AC9.1.5: Sistema registra >= 1 falso-positivo documentado para validação

**INVEST Notes**: Independent (after US7.1), Negotiable (thresholds adjustable), Valuable (risk mitigation), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 9: Integração CRM HubSpot (FR11)

### US11.1 — Sincronizar Lead Qualificado
**Como** Agente SDR, **quero** sincronizar lead qualificado com o CRM, **para que** o lead fique na esteira Kanban e possa ser acompanhado pela equipe comercial.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.2

**Acceptance Criteria**:
- AC11.1.1: MCP HubSpot numa demonstração única ao vivo (usando MCP auth app + MCP Inspector)
- AC11.1.2: CRM-adapter (SQS→Lambda) com simulado default + Private App REST
- AC11.1.3: Sistema sincroniza lead qualificado e status da esteira Kanban
- AC11.1.4: Sistema registra a rota do lead no CRM para auditoria
- AC11.1.5: Integração não bloqueia o fluxo principal se falhar (DLQ)

**INVEST Notes**: Independent (after US3.2), Negotiable (fallback to simulated acceptable), Valuable (CRM integration), Estimable (clear), Small (focused), Testable (verifiable)

---

## Nice-to-Have Stories (se sobrar tempo)

### US10.1 — Roleta de Distribuição (FR10)
**Como** Agente SDR, **quero** distribuir leads qualificados para corretores via regras configuráveis, **para que** leads grandes vão para especialistas e leads menores para rodízio.

**Priority**: Should Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.2

---

### US8.1 — Follow-up Automático (FR8)
**Como** Lead B2B, **quero** receber follow-up automático se eu parar de responder, **para que** eu não perca oportunidades por falta de acompanhamento.

**Priority**: Should Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.2

---

### US12.1 — Ingestão de E-mail (FR12)
**Como** Lead B2B, **quero** que minha solicitação via portal/e-mail inicie automaticamente uma conversa com o agente, **para que** eu não precise digitar manualmente minhas informações.

**Priority**: Could Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: Nenhuma

---

## Notas de Implementação

- Stories seguem priorização MoSCoW alinhada com must-have vs nice-to-have dos requisitos
- Stories são pequenas (1-3 dias) para melhor track de progresso
- Critérios de aceitação são testáveis e mensuráveis
- Traceabilidade com FRs será mantida via traceability.json