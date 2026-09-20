# Architecture Decision Records — Agente SDR Imobiliário B2B

> Estágio Domain Design (Inception). Registro de decisões de arquitetura significativas.

---

## ADR-001: Núcleo Síncrono como Módulos Internos de Uma Lambda

**Context**
O sistema precisa responder a mensagens do Telegram em tempo real (just-in-time da mentoria: leads de portal "queimam rápido"). Lambda→Lambda síncrono é antipattern (custo dobrado, timeout em cascata, acoplamento). Múltiplas Lambdas síncronas aumentariam latência e complexidade para a POC.

**Decision**
Os componentes síncronos (SalesFlow, SecurityLayer, SDR Agent, LeadQualifier, PropertiesRAG, Scheduler, Handoff, LeadRouter) rodam como módulos/bibliotecas internos de uma única Lambda (ConversationRouter). Apenas componentes assíncronos (VoiceAdapter, CRMAdapter, ContactIngest, AnomalyDetector, Followup, DashAPI) são Lambdas próprias, acionadas por SQS/SES/EventBridge.

**Consequences**
**Positivos:**
- Latência mínima para resposta ao lead (sem cold chain de Lambdas)
- Custo reduzido (uma invocação Lambda ao invés de múltiplas)
- Simplicidade para POC (menos infraestrutura)
- Evita timeout em cascata

**Negativos:**
- Tamanho do pacote da Lambda aumenta (todos os módulos internos)
- Deploy de módulo interno requer redeploy da Lambda inteira
- Escalabilidade vertical (mais memória) ao invés de horizontal

**Alternatives Rejected**
- **Lambda→Lambda síncrono**: Rejeitado por custo dobrado, timeout em cascata, acoplamento
- **Event-driven total**: Rejeitado porque chat exige latência mínima (just-in-time da mentoria)
- **Monolith único**: Rejeitado porque componentes assíncronos devem ser Lambdas próprias (transcrição de áudio lenta, CRM síncrono bloqueante)

---

## ADR-002: Separação de SalesFlow e SDR Agent

**Context**
O PRD descreve SalesFlow como engine de fluxo conversacional (LangGraph) e SDR Agent como gerador de respostas via LLM. É possível combinar ambos em um único componente.

**Decision**
SalesFlow e SDR Agent são componentes separados. SalesFlow gerencia o grafo de estados conversacionais (saudação, elicitação, intenção, qualificação, recomendação, agendamento, follow-up, handoff). SDR Agent gera respostas humanizadas via LLM (OpenRouter/Claude 3.5 Haiku) e é chamado por SalesFlow em cada estado.

**Consequences**
**Positivos:**
- Separação de concerns (orquestração vs geração de resposta)
- SalesFlow pode ser testado independentemente do LLM
- SDR Agent pode ser substituído por outro provedor LLM sem alterar SalesFlow
- Alinhado com padrão LangGraph (agent nodes separados do grafo)

**Negativos:**
- Leve overhead de chamada entre componentes
- Complexidade adicional (mais componentes)

**Alternatives Rejected**
- **SalesFlow + SDR Agent combinados**: Rejeitado porque mistura orquestração de estados com geração de resposta, violando separação de concerns

---

## ADR-003: PropertiesRAG com FAISS Local em Memória

**Context**
O PRD exige RAG sobre base de imóveis (100–200 ofertas). Opções incluem FAISS local (carrega em memória), OpenSearch Serverless, ou Bedrock Knowledge Bases.

**Decision**
Na POC, usar FAISS local com índice carregado em memória da Lambda. Arquitetura permite troca por Bedrock Knowledge Bases + OpenSearch Serverless sem alterar contrato. Base de imóveis é sintética e calibrada por fontes públicas (FipeZAP, Secovi-SP, GeoSampa).

**Consequences**
**Positivos:**
- Custo zero para POC (sem OpenSearch Serverless)
- Latência mínima (índice em memória)
- Simplicidade (sem serviço externo)

**Negativos:**
- Índice limitado pelo tamanho da memória da Lambda
- Escalabilidade limitada (se base crescer significativamente)
- Reconstrução de índice requer redeploy

**Alternatives Rejected**
- **OpenSearch Serverless**: Rejeitado por custo para POC
- **Bedrock Knowledge Bases**: Rejeitado por custo e complexidade para POC
- **RAG externo (Pinecone, etc.)**: Rejeitado por custo e dependência de terceiros

---

## ADR-004: AnomalyDetector como Job Diário (Lambda Própria)

**Context**
O PRD exige detecção de anomalias em conversas (módulo da fase). Opções incluem monitor em tempo real (dentro do fluxo conversacional) ou job diário independente.

**Decision**
AnomalyDetector é uma Lambda própria acionada por EventBridge como job diário. Extrai features por conversa (volume, comprimento, sentimento, horários atípicos) e aplica Isolation Forest + PCA + Autoencoder. Emite alerta no dashboard e restringe agendamento para leads suspeitos.

**Consequences**
**Positivos:**
- Não impacta latência do fluxo conversacional
- Pode processar todas as conversas de um dia em batch
- Algoritmos complexos (Isolation Forest + PCA + Autoencoder) têm tempo de processamento adequado
- Alertas são proativos (dashboard atualizado diariamente)

**Negativos:**
- Detecção não é em tempo real (até 24h de delay)
- Lead suspeito pode continuar conversando antes de ser bloqueado

**Alternatives Rejected**
- **Monitor em tempo real**: Rejeitado porque impactaria latência do fluxo conversacional e algoritmos complexos não seriam viáveis em tempo real

---

## ADR-005: Dashboard como Streamlit Community Cloud (Fora da AWS)

**Context**
O PRD exige dashboard mínimo com KPIs e anomalias. Opções incluem Streamlit Community Cloud (grátis), S3 estático + HTML/JS (100% AWS), ou Lambda + API Gateway.

**Decision**
Dashboard é app Streamlit (1 página) no Community Cloud (grátis). Consome GET /api/kpis (Lambda DashAPI + DynamoDB/CloudWatch). Login via Cognito. Única peça fora da AWS.

**Consequences**
**Positivos:**
- Custo zero (Community Cloud é grátis)
- Desenvolvimento rápido (Python, não HTML/JS)
- Simplicidade (Streamlit abstrai UI)

**Negativos:**
- Fora da AWS (não é 100% serverless)
- Comunidade Cloud pode ter limites de uso
- Não é enterprise-ready (produção exigiria hosting próprio)

**Alternatives Rejected**
- **S3 estático + HTML/JS**: Rejeitado por exigir muito mais front-end para o mesmo resultado
- **Lambda + API Gateway**: Rejeitado por complexidade e custo para POC

---

## ADR-006: CRMAdapter com MCP e Simulado

**Context**
O PRD exige integração com CRM (HubSpot/Kenlo/Facilita). CRMs imobiliários brasileiros não têm MCP nativo. Opções incluem integração direta com API REST ou camada MCP genérica.

**Decision**
CRMAdapter é uma camada MCP genérica que conecta a qualquer servidor MCP de CRM (HubSpot) ou envolve API REST de CRM imobiliário (Kenlo/Facilita) num servidor MCP próprio. Na POC, roda contra CRM simulado (CSV/Excel). Fluxo do agente fica desacoplado do vendedor.

**Consequences**
**Positivos:**
- Fluxo do agente desacoplado do CRM específico
- "Pronto para conectar" sem depender de CRM externo real
- Demo única ao vivo com HubSpot (MCP) é possível
- Troca de CRM requer apenas adaptação do adapter

**Negativos:**
- Camada adicional (MCP) adiciona complexidade
- MCP não é nativo para CRMs imobiliários brasileiros

**Alternatives Rejected**
- **Integração direta com API REST de CRM específico**: Rejeitado porque acopla fluxo ao vendedor e não é "pronto para conectar"

---

## ADR-007: VoiceAdapter como Lambda Própria (SQS)

**Context**
O PRD exige transcrição de voz (faster-whisper PT-BR). Transcrição é lenta e não deve bloquear resposta ao lead. Opções incluem processamento síncrono ou assíncrono.

**Decision**
VoiceAdapter é uma Lambda própria acionada por SQS. ConversationRouter enfileira mensagens de áudio. VoiceAdapter baixa arquivo, converte para WAV (ffmpeg), transcreve com faster-whisper, e envia texto transcrito de volta.

**Consequences**
**Positivos:**
- Não bloqueia resposta ao lead
- Transcrição lenta não impacta latência
- SQS com DLQ garante resiliência

**Negativos:**
- Lead não recebe resposta imediata sobre áudio
- Delay entre envio de áudio e resposta transcrita

**Alternatives Rejected**
- **Processamento síncrono**: Rejeitado porque transcrição é lenta e bloquearia resposta ao lead

---

## ADR-008: SecurityLayer como Módulo Interno (PII Masking)

**Context**
O PRD exige LGPD compliance com PII masking. Opções incluem masking antes do LLM (módulo interno) ou masking via provedor LLM (Bedrock Guardrails).

**Decision**
SecurityLayer é um módulo interno de ConversationRouter. Extrai e persiste PII real (nome, e-mail, telefone, CNPJ) no DynamoDB criptografado (KMS). Substitui PII por placeholders no texto enviado ao LLM. Valida saída contra vazamento de PII (regex). Independente do provedor LLM (OpenRouter ou Bedrock).

**Consequences**
**Positivos:**
- PII nunca chega ao provedor LLM (independente do provedor)
- Compliance LGPD garantido em código
- Minimização de dados (só necessário é enviado)

**Negativos:**
- Complexidade adicional (extraction, masking, validation)
- Overhead de processamento

**Alternatives Rejected**
- **Masking via provedor LLM (Bedrock Guardrails)**: Rejeitado porque acopla a Bedrock e não garante compliance com OpenRouter (POC usa OpenRouter)

---

## ADR-009: LeadRouter com Regras Configuráveis

**Context**
A mentoria deixou explícito que lead de alto ticket (> 500 m²) deve ir para diretor/especialista, enquanto lead menor vai para rodízio de consultores. Opções incluem regras hard-coded ou configuráveis.

**Decision**
LeadRouter aplica regras configuráveis (ex.: até 500 m² → rodízio dos consultores; acima → diretor/especialista). Regras são armazenadas em configuração (pode ser DynamoDB ou ambiente). Registra a rota no DynamoDB para auditoria.

**Consequences**
**Positivos:**
- Flexibilidade (regras podem ser ajustadas sem código)
- Alinhado com insight da mentoria
- Auditoria completa da rota

**Negativos:**
- Complexidade adicional (configuração de regras)
- Necessidade de validação de regras

**Alternatives Rejected**
- **Regras hard-coded**: Rejeitado porque não permite ajuste sem código e não é alinhado com insight da mentoria