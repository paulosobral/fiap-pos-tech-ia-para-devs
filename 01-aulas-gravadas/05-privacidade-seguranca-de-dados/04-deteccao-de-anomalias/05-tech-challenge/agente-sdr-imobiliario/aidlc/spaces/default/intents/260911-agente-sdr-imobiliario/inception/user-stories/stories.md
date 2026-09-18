# User Stories — Agente SDR Imobiliário B2B

## Grupo 1: Atendimento Conversacional (FR1)

### US1.1 — Iniciar Conversa
**Como** Lead B2B, **quero** iniciar uma conversa com o agente SDR via Telegram, **para que** eu possa obter informações sobre imóveis corporativos sem esperar horário comercial.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: Nenhuma

**Acceptance Criteria**:
- AC1.1.1: Lead envia `/start` ou primeira mensagem no Telegram
- AC1.1.2: Agente responde em < 10s (p90, carga e hardware definidos, amostra mínima) com mensagem inicial padrão
- AC1.1.3: Sistema cria sessão no DynamoDB com ID único
- AC1.1.4: Lead recebe consentimento LGPD contextualizado na primeira mensagem (finalidade, dados coletados, retenção 90 dias, compartilhamento com corretor/CRM, como recusar/revogar)
- AC1.1.5: Comportamento definido quando consentimento é recusado (não coletar, encerrar educadamente)
- AC1.1.6: Mensagens duplicadas/infila de webhook são deduplicadas (idempotência)

**INVEST Notes**: Independent (não depende de outras stories), Negotiable (detalhes da saudação podem ajustar), Valuable (first touchpoint), Estimable (clear boundary), Small (focused), Testable (clear acceptance criteria)

---

### US1.2 — Conversar com Texto Livre
**Como** Lead B2B, **quero** digitar mensagens de texto livre para o agente, **para que** eu possa expressar minhas necessidades de forma natural sem seguir um menu rígido.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US1.1

**Acceptance Criteria**:
- AC1.2.1: Agente aceita texto livre e o processa conforme os intents esperados (saudação, busca, agendamento, dúvida)
- AC1.2.2: Sistema mantém contexto conversacional entre mensagens
- AC1.2.3: Agente oferece botões inline como atalhos (mas não exige)
- AC1.2.4: Contexto é persistido no DynamoDB com TTL 90 dias (política de expiração verificável)
- AC1.2.5: Mensagens vazias, idioma não suportado e texto muito longo têm resposta de fallback definida

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
- AC1.3.4: Texto transcrito entra no fluxo conversacional como mensagem digitada; ao usuário é mostrado o texto transcrito para revisão antes de ações sensíveis
- AC1.3.5: Transcrição acontece em < 15s (p90, amostra mínima, condições de carga/hardware definidas) para áudios < 30s, com feedback imediato de processamento e fallback assíncrono quando exceder limite
- AC1.3.6: Decisão de POC: **voice-in/text-out** — sem TTS/retorno por voz nesta fase
- AC1.3.7: Áudios inválidos, sem fala, ruído, idioma diferente ou falha de ffmpeg/STT têm fallback definido (pedir texto/retry sem perder contexto)

**INVEST Notes**: Independent (after US1.1), Negotiable (timeout acceptable), Valuable (convenience), Estimable (clear technical path), Small (focused), Testable (verifiable)

---

## Grupo 2: Identificação de Intenção (FR2)

### US2.1 — Detectar Intenção
**Como** Agente SDR, **quero** identificar a intenção do lead (compra/locação/investimento) no início da conversa, **para que** eu possa adaptar o fluxo de qualificação apropriado.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US1.2

**Acceptance Criteria**:
- AC2.1.1: Sistema classifica intenção com >= 85% de precisão — protocolo de medição fixado: dataset rotulado definido (n-classes), tamanho mínimo de amostra, métrica (precisão macro), split treino/avaliação, regra para baixa confiança
- AC2.1.2: Classificação acontece nas primeiras 3 trocas de mensagem
- AC2.1.3: Lead B2B é diferenciado de Investidor PF automaticamente (classificação de persona separada da de intenção)
- AC2.1.4: Sistema registra intenção detectada no DynamoDB
- AC2.1.5: Casos de baixa confiança/ambiguidade têm regra definida (pedir confirmação sem quebrar o fluxo)

**INVEST Notes**: Independent (after US1.2), Negotiable (3 turns acceptable), Valuable (routing accuracy), Estimable (clear metric), Small (focused), Testable (verifiable)

---

### US2.2 — Coletar Informações Básicas
**Como** Agente SDR, **quero** coletar informações básicas do lead (metragem, região, orçamento, prazo, nº pessoas, decisor), **para que** eu possa qualificar o lead adequadamente.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US2.1

**Acceptance Criteria**:
- AC2.2.1: Sistema aplica questionário adaptativo baseado em intenção detectada
- AC2.2.2: Para compra/locação: coleta metragem, região, orçamento, prazo, nº pessoas, decisor (campos e validações de unidade/limite explícitos)
- AC2.2.3: Para investimento: coleta ticket médio, expectativa de retorno (%), período
- AC2.2.4: Informações são estruturadas e persistidas no DynamoDB
- AC2.2.5: Questionário não repete perguntas já respondidas; permite correção de resposta, "não sei" e retomada após interrupção

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
- AC3.1.2: Score vai de 0-100, onde >= 70 indica lead quente (fórmula/versão e pesos documentados; limites inclusivos definidos)
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
- AC3.2.1: Sistema propõe handoff quando score >= 70 OU lead solicita (precedência definida)
- AC3.2.2: Sistema coleta informações restantes obrigatórias por persona/intenção antes do handoff
- AC3.2.3: Lead recebe confirmação antes do handoff, informando o que será compartilhado e o prazo/canal de retorno esperado
- AC3.2.4: Sistema bloqueia handoff de leads com anomalias detectadas, com mensagem neutra e caminho para atendimento humano (sem revelar lógica antifraude)
- AC3.2.5: Handoff é idempotente (não gera duplicatas em falha/retry)

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
- AC4.1.3: Sistema nunca inventa imóveis que não estão na base (constraint via RAG); cada recomendação referencia identificador existente
- AC4.1.4: Índice FAISS e modelo de embeddings carregados (definir estratégia de empacotamento/limite da Lambda; comportamento cold/warm start)
- AC4.1.5: Busca acontece em < 2s (p90, amostra definida)
- AC4.1.6: Sem resultados: apresenta alternativa útil e permite ajustar filtros

**INVEST Notes**: Independent (after US2.2), Negotiable (k-value adjustable), Valuable (relevant recommendations), Estimable (clear performance), Small (focused), Testable (verifiable)

---

### US4.2 — Apresentar Opções
**Como** Lead B2B, **quero** ver até 3 opções de imóveis com detalhes relevantes, **para que** eu possa comparar e escolher a melhor opção.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US4.1

**Acceptance Criteria**:
- AC4.2.1: Sistema apresenta até 3 imóveis com área útil, área bruta, condomínio, localização, data de atualização, disponibilidade e unidade monetária
- AC4.2.2: Cada opção diferencia claramente compra vs locação vs investimento
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
- AC5.1.1: Sistema valida data/hora disponível no calendário simulado (timezone America/Sao_Paulo, duração, antecedência mínima)
- AC5.1.2: Sistema grava compromisso no calendário simulado de forma idempotente (sem dupla reserva)
- AC5.1.3: Sistema emite convite ICS para o lead
- AC5.1.4: Sistema notifica corretor via canal interno (Telegram/e-mail)
- AC5.1.5: Lead recebe confirmação do agendamento
- AC5.1.6: Falha parcial (calendário/ICS/notificação) resulta em estado de erro recuperável e permite remarcar/cancelar

**INVEST Notes**: Independent (after US4.2), Negotiable (validation logic flexible), Valuable (convenience), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 6: Resumo Inteligente (FR6)

### US6.1 — Gerar Handoff
**Como** Corretor Especialista, **quero** receber um resumo inteligente do lead qualificado, **para que** eu possa ter contexto completo antes de iniciar o contato sem ter que ler toda a conversa.

**Priority**: Must Have  
**Persona**: Corretor Especialista (interna)  
**Depends**: US3.2

**Acceptance Criteria**:
- AC6.1.1: Sistema gera handoff em Markdown com schema mínimo definido (score, intenção, urgência, dados coletados, próximos passos, assumido/faltante)
- AC6.1.2: Sistema envia mensagem interna + arquivo de resumo ao corretor
- AC6.1.3: Sistema desmascara PII apenas no handoff interno (criptografia e fronteira de acesso/papel); LLM nunca recebe PII bruta no contexto
- AC6.1.4: Handoff inclui todas as informações coletadas (metragem, região, orçamento, etc.)
- AC6.1.5: Handoff é gerado automaticamente quando lead atinge score >= 70 (disparo idempotente, sem duplicatas)
- AC6.1.6: Handoff ausente/incompleto não gera duplicação; dados sensíveis não aparecem em logs/trace

**INVEST Notes**: Independent (after US3.2), Negotiable (format flexible), Valuable (efficiency), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 7: Dashboard Mínimo (FR7)

### US7.1 — Monitorar KPIs
**Como** Gestor W Levitt, **quero** ver métricas de operação em tempo real, **para que** eu possa tomar decisões informadas sobre investimentos e operação.

**Priority**: Must Have  
**Persona**: Gestor W Levitt  
**Depends**: Nenhuma (paralelo em UI; dados dependem de eventos produzidos por US1.1, US3.1, US5.1, US9.1)

**Acceptance Criteria**:
- AC7.1.1: Dashboard consome `GET /api/kpis` com métricas de negócio (período e timezone definidos)
- AC7.1.2: Dashboard exibe linha de métricas: leads hoje/semana, tempo 1ª resposta p90, taxa qualificação, agendamentos
- AC7.1.3: Dashboard exibe gráficos: volume de intenções, leads distribuídos pela roleta
- AC7.1.4: Dashboard exibe tabela de anomalias com alertas 24h (status, severidade, timestamp, explicação acionável)
- AC7.1.5: Dashboard é acessível via login Cognito (JWT) com autorização por perfil e expiração/renovação

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
- AC9.1.2: Sistema aplica Isolation Forest + PCA + Autoencoder (versão/configuração dos modelos e regra de combinação definidas; dataset de calibração registrado)
- AC9.1.3: Sistema emite alerta no dashboard quando anomalia é detectada (evita alerta duplicado; distingue novo/em investigação/falso positivo/resolvido)
- AC9.1.4: Sistema restringe agendamento para leads suspeitos com mensagem compreensível e opção de atendimento humano (sem expor lógica antifraude)
- AC9.1.5: Sistema registra falso-positivo documentado (dataset, método e resultado de validação)
- AC9.1.6: Tratado como spike técnico de calibração + story de produto (escopo maior que 1-3 dias)

**INVEST Notes**: Independent (after US7.1), Negotiable (thresholds adjustable), Valuable (risk mitigation), Estimable (clear), Small (focused), Testable (verifiable)

---

## Grupo 9: Integração CRM HubSpot (FR11)

### US11.1 — Sincronizar Lead Qualificado
**Como** Agente SDR, **quero** sincronizar lead qualificado com o CRM, **para que** o lead fique na esteira Kanban e possa ser acompanhado pela equipe comercial.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.2

**Acceptance Criteria**:
- AC11.1.1: MCP HubSpot demonstrado com critérios verificáveis (payload, autenticação, lead/status sincronizados, correlação/auditoria) — sem depender de demonstração única ao vivo
- AC11.1.2: CRM-adapter (SQS→Lambda) com simulado default + Private App REST
- AC11.1.3: Sistema sincroniza lead qualificado e status da esteira Kanban
- AC11.1.4: Sistema registra a rota do lead no CRM para auditoria
- AC11.1.5: Integração não bloqueia o fluxo principal se falhar (DLQ); retries, replay e idempotência definidos

**INVEST Notes**: Independent (after US3.2), Negotiable (fallback to simulated acceptable), Valuable (CRM integration), Estimable (clear), Small (focused), Testable (verifiable)

---

### US13.1 — Meta de Qualificação da POC (≥ 60%)
**Como** Gestor W Levitt, **quero** confirmar que a operação atinge a meta de pelo menos 60% de leads qualificados, **para que** eu possa validar o valor do agente na POC.

**Priority**: Must Have  
**Persona**: Gestor W Levitt  
**Depends**: US3.2, US7.1

**Acceptance Criteria**:
- AC13.1.1: Taxa de qualificação calculada como `leads com score >= 70 OU handoff` ÷ `leads ativos no período` (denominador explícito: leads com NOVA sessão no período)
- AC13.1.2: Janela de medição definida (ex.: 7 dias corridos após go-live da POC)
- AC13.1.3: Dashboard exibe a taxa de qualificação com período e denominador visíveis
- AC13.1.4: Meta >= 60% aferida sobre o período com evidência (export dos KPIs)

**INVEST Notes**: Estimable (clear metric), Testable (measurable protocol)

---

## Grupo 10: Stories Técnicas / Cobertura NFR (NFR1–NFR9)

Stories transversais que cobrem os NFRs sem artifacts de negócio próprios; critérios mensuráveis substituem métricas subjetivas.

### US14.1 — Segurança de Dados e Privacidade (NFR2/NFR3/NFR4/NFR7)
**Como** Agente SDR, **quero** aplicar mascaramento de PII, criptografia, retenção e auditoria em todo o fluxo, **para que** a operação cumpra LGPD e proteção de dados.

**Priority**: Must Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US1.1

**Acceptance Criteria**:
- AC14.1.1: PII é mascarada antes de qualquer chamada ao LLM (payload contém apenas dados não sensíveis)
- AC14.1.2: PII criptografada em repouso (KMS); chaves versionadas e rotacionáveis
- AC14.1.3: Retenção de 90 dias (DynamoDB TTL) é auditável e verificável por política
- AC14.1.4: Consentimento, revogação, consulta, correção e exclusão de dados são suportados, com evidência no audit log
- AC14.1.5: Logs e traces não contêm PII bruta nem tokens

### US15.1 — Observabilidade e Monitoramento (NFR5/NFR6)
**Como** Gestor W Levitt, **quero** logs, métricas e alertas operacionais da conversa e integrações, **para que** eu possa diagnosticar falhas e custos.

**Priority**: Should Have  
**Persona**: Gestor W Levitt  
**Depends**: US1.1

**Acceptance Criteria**:
- AC15.1.1: Logs estruturados de fluxo (sessão, intenção, score, handoff, anomalia) com IDs de correlação
- AC15.1.2: Métricas de latência, erro, volume e custo por invocação/LLM expostas (CloudWatch)
- AC15.1.3: Alertas configurados para taxa de erro, latência p90 e falha de integração (DLQ não-vazia)
- AC15.1.4: Traces de ponta a ponta com fuso/UUID de correlação entre Telegram ↔ Lambda ↔ DynamoDB ↔ integrações

### US16.1 — Escalabilidade, Custo e IaC (NFR8/NFR9)
**Como** Agente SDR, **quero** provisionar e gerenciar o ambiente por infraestrutura como código com limites de custo, **para que** a POC seja recriável e financeiramente controlada.

**Priority**: Should Have  
**Persona**: Gestor W Levitt  
**Depends**: Nenhuma (paralelo)

**Acceptance Criteria**:
- AC16.1.1: Todo provisionamento via IaC (Terraform) recriável com `start.sh`/`stop.sh` e `terraform apply`/`destroy` (prática de equipe)
- AC16.1.2: Orçamento/alerta de custo configurado (billing alert); consumos por serviço estimados antes do deploy
- AC16.1.3: Concorrência e escala: sistema com TTL, DLQ e throttling; processo pesado (STT, anomalia) fora do caminho síncrono da Lambda
- AC16.1.4: Testes de jornada core executam no ambiente provisionado (smoke test)

**INVEST Notes**: Stories técnicas transversais — não são stories de usuário; track separado no Delivery Planning.

## Nice-to-Have Stories (se sobrar tempo)

### US10.1 — Roleta de Distribuição (FR10)
**Como** Agente SDR, **quero** distribuir leads qualificados para corretores via regras configuráveis, **para que** leads grandes vão para especialistas e leads menores para rodízio.

**Priority**: Should Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.2

**Acceptance Criteria**:
- AC10.1.1: Regras de distribuição são configuráveis e versionadas (área, especialidade, disponibilidade, capacidade)
- AC10.1.2: Sistema aplica lock/idempotência para evitar distribuição duplicada do mesmo lead
- AC10.1.3: A decisão da roleta é registrada e auditável
- AC10.1.4: Comportamento definido para empate, sem corretor disponível (fallback)

---

### US8.1 — Follow-up Automático (FR8)
**Como** Lead B2B, **quero** receber follow-up automático se eu parar de responder, **para que** eu não perca oportunidades por falta de acompanhamento.

**Priority**: Should Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: US3.2

**Acceptance Criteria**:
- AC8.1.1: Sistema agenda follow-up (EventBridge) na cadência dia 2/5/9 após inatividade
- AC8.1.2: Follow-up respeita janela de silêncio (horários definidos)
- AC8.1.3: Follow-up é cancelado quando o lead responde
- AC8.1.4: Follow-up usa contexto/memória da conversa anterior

---

### US12.1 — Ingestão de E-mail (FR12)
**Como** Lead B2B, **quero** que minha solicitação via portal/e-mail inicie automaticamente uma conversa com o agente, **para que** eu não precise digitar manualmente minhas informações.

**Priority**: Could Have  
**Persona**: Lead B2B, Investidor PF  
**Depends**: Nenhuma

**Acceptance Criteria**:
- AC12.1.1: Sistema recebe e-mail (SES), faz parsing e validação do conteúdo
- AC12.1.2: Sistema deduplica e-mails repetidos
- AC12.1.3: Sistema abre sessão equivalente a US1.1 com as informações extraídas do e-mail

---

## Notas de Implementação

- Stories seguem priorização MoSCoW alinhada com must-have vs nice-to-have dos requisitos
- Stories são pequenas (1-3 dias) para melhor track de progresso; US9.1 é exceção (spike de calibração + implementação)
- Critérios de aceitação são testáveis e mensuráveis, com percentil/amostra quando há meta de performance
- Traceabilidade com FRs será mantida via traceability.json (também mapear NFRs → stories → ACs)
- Cross-cutting (IaC, observabilidade, segurança, testes, deploy) é camada transversal; considerar como stories técnicas separadas no Delivery Planning
- Acessibilidade é requisito transversal: texto alternativo, navegação por teclado, contraste, não depender apenas de cor/áudio/emoji, screen reader no dashboard
- WhatsApp fora do escopo da POC; manter contrato de canal agnóstico no desenho
- Decisão de voz na POC: voice-in/text-out (ver US1.3)