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

**Atualização (2026-10-09)**
Na implementação o núcleo síncrono roda como **serviço ECS Fargate** (task `sdr-conversation-router`, `server.py` com `http.server`, porta 8080), não como Lambda. O API Gateway o alcança por `HTTP_PROXY` para o IP da task (`router_endpoint`), que o `start.sh` atualiza depois de ligar o serviço; o serviço liga às 09:00 e desliga às 18:00 (BRT). A decisão de manter os módulos internos de um único serviço, sem chamada Lambda→Lambda, continua válida. O motivo da troca de Lambda para ECS não foi registrado na época.

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

**Atualização (2026-10-09)**
- A varredura roda pelo EventBridge a cada minuto (`anomaly_schedule = rate(1 minute)`), não uma vez por dia; o `anomaly_id` determinístico (`sessão#data`) mantém o reprocessamento idempotente.
- O scorer padrão em produção é o **heurístico** (`ANOMALY_SCORER=heuristic`, limite 0,7); Isolation Forest + PCA é opcional (`sklearn`). O **Autoencoder não foi implementado** (desvio aceito, FR9.2).
- São 4 features: volume, tamanho médio, proporção de mensagens negativas e proporção fora do horário comercial. Ver ADR-019 para o gate de qualidade ponta a ponta.

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

**Atualização (2026-10-09)**
O dashboard **não** roda no Streamlit Community Cloud: é um container no **ECS Fargate** (janela 09:00–18:00 BRT), sem domínio nem ALB, e por isso sem `st.login`/Hosted UI (ADR-014). Ver também ADR-016 (leads e sessão) e ADR-018 (identidade visual).

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

**Atualização (2026-10-09)**
O `voice-adapter` é um **worker ECS Fargate** que consome a fila SQS de áudio (long-polling) com `faster-whisper` + `ffmpeg`, não uma Lambda: o pacote (~520 MB descompactado) não cabe no limite do Lambda. Liga e desliga na mesma janela 09:00–18:00 BRT; fora dela o áudio fica na fila. A transcrição entra no fluxo como se o lead tivesse digitado, mantendo o aviso "coloquei na fila para transcrição" (ADR-015).

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

## ADR-010: Roteamento de Modelos LLM em Camadas (Tier 1 Flash + Tier 2 Complex/Fallback) com LiteLLM e SSM

**Context**
Modelos premium (ex.: Claude 3.5 Sonnet / Claude 3.5 Haiku) têm custo desnecessariamente elevado para tarefas corriqueiras do SDR (80% do tráfego: saudações, triagem, elicitação e perguntas simples). Além disso, falhas de provedor (HTTP 429, timeouts, rate limits) quebram a experiência do usuário caso não haja mecanismo automático de contingência.

**Decision**
Implementar uma arquitetura de LLM em 3 camadas orquestrada via LiteLLM e AWS SSM Parameter Store:
1. **Tier 1 (Rotina - 90% das chamadas)**: modelo econômico e veloz (`deepseek/deepseek-chat` ou equivalente Flash) para classificação de intenção, triagem e polimento de respostas de qualificação.
2. **Tier 2 (Fallback Automático de Erro)**: fallback resiliente (`anthropic/claude-3-haiku` ou fallback model) acionado transparentemente pelo LiteLLM em caso de timeout, 429 ou erro do provedor primário.
3. **Tier 3 (Complex / Handoff / Argumentação Avançada)**: modelo de alta capacidade (`anthropic/claude-3.5-sonnet`) acionado condicionalmente em fluxos de negociação sofisticada ou dúvidas consultivas complexas.
Parâmetros dinâmicos gerenciados no AWS SSM Parameter Store (`/sdr/llm-model-primary`, `/sdr/llm-model-fallback`, `/sdr/llm-model-complex`).

**Consequences**
**Positivos:**
- Redução de ~75% a 90% do consumo de tokens para conversas padrão.
- Alta resiliência (zero downtime em 429/indisponibilidade via fallback nativo).
- Troca a quente de provedores via Parameter Store sem novo deploy de imagem.

**Negativos:**
- Ligeiro aumento na complexidade de configuração e gestão de múltiplos parâmetros no SSM.
- Necessidade de testes de integração cobrindo os caminhos de fallback.

**Atualização (2026-10-09)**
Os modelos Tier 2 e Tier 3 citados acima (`anthropic/claude-3-haiku` e `anthropic/claude-3.5-sonnet`) foram **descontinuados no OpenRouter** (a chamada devolve 404). Valores em vigor: Tier 1 `deepseek/deepseek-chat`, Tier 2 `anthropic/claude-haiku-4.5` e Tier 3 `anthropic/claude-sonnet-4.5`, definidos nas variáveis do Terraform e gravados nos parâmetros SSM `/sdr/llm-model-primary|fallback|complex`; também podem ser sobrescritos por `LLM_MODEL_PRIMARY|FALLBACK|COMPLEX`. Em 08/10/2026 o Tier 1 devolveu 429 do provedor (limite upstream) em vários turnos; nesse caso o cliente cai para o Tier 2, ao custo de alguns segundos a mais por turno.

## ADR-011: Roteamento Conversacional Agentic via LangGraph (LLM decide transição, gates de negócio ficam em código)

**Context**
A extração de dados do lead (`area`, `region`, `budget`, `deadline`, `people_count`, `decision_maker`) e a detecção de intenções contextuais ("cadê as opções?", "tem mais opções?", recusa a meio de fluxo) eram feitas 100% via regex hardcoded, uma expressão por variação de frase (`_AREA_RE`, `_REGION_RE`, `_BUDGET_PREFIXED_RE`, `_BUDGET_CEILING_RE`, `_DECISOR_YES_RE`, `_OPTIONS_REQUEST_RE`, entre outras). Cada nova forma de o usuário se expressar (ex.: "em Pinheiros" sem prefixo "bairro/região", "quem decide sou eu", "até 5000 reais", "tem mais opções?" durante o estado `scheduling`) exigia um regex novo, revelando bugs recorrentes de robustez de linguagem natural e tornando o `LangGraph` um roteador decorativo (`_route_state` apenas lia `current_state`, sem decisão real). A análise competitiva (`ideation/market-research/competitive-analysis.md`) registra a **qualificação com score explicável** como diferencial sustentável da POC frente a Lais.ai/Maya — esse diferencial não pode ser perdido ao evoluir a robustez conversacional.

**Decision**
Inserir um nó `router` acionado por LLM entre `preprocess` e os nós de destino do grafo, com decisão restrita a um enum de transições válidas computado em código:
1. **Extração estruturada via LLM** (`service/llm.py::extract_and_route`, Tier 1 do ADR-010): uma chamada LLM por turno retorna JSON com os deltas de `lead_info` extraídos da mensagem livre do usuário, substituindo os regex de extração (que passam a existir apenas como fallback de segurança).
2. **Gates de negócio permanecem 100% em código**, nunca delegados ao LLM: consentimento LGPD obrigatório antes de `intent`; `LeadQualifier.calculate_score()` (inalterado, determinístico) com `SCORE_THRESHOLD` obrigatório antes de `recommendation`; verificação de restrição de agendamento (`_is_restricted`); recusa ("não") sempre terminal.
3. **Nó `router` (LLM)** recebe a mensagem, o histórico relevante, `lead_info` e o **enum de transições válidas** (calculado por `compute_valid_transitions(state)`, função pura em código que aplica os gates do item 2) e escolhe uma delas. Transição fora do enum ou falha do LLM → fallback determinístico para o FSM regex atual (nunca bloqueia o turno).
4. Reaproveita o roteamento em camadas e o fallback automático do ADR-010 para a chamada de `extract_and_route`.

**Consequences**
**Positivos:**
- Elimina a classe de bug "frase nova = regex novo": qualquer forma de o usuário pedir mais opções, recusar, ou fornecer dados generaliza via LLM, sem lista de regex crescente.
- Mantém o score de qualificação 100% determinístico e testável sem LLM — preserva o diferencial competitivo declarado.
- LangGraph passa a ter decisão real na aresta condicional (`router`), alinhado ao padrão de mercado de agentes conversacionais.
- Fallback determinístico existente (regex + FSM) não é descartado — vira rede de segurança para falha/timeout do LLM.

**Negativos:**
- Uma chamada LLM adicional por turno (latência ~200-400ms), mitigada por reuso do Tier 1 econômico (ADR-010).
- Superfície de teste maior — requer mocks de `llm_router` (mesmo padrão de DI de `llm_classify_intent`/`reply_generator`) para manter testes determinísticos e sem chamada real de API.

**Alternatives Rejected**
- **Delegar o score de qualificação ao LLM**: rejeitado por eliminar a explicabilidade/auditabilidade que é diferencial competitivo (ver Q1 da pergunta de brainstorming ao usuário, resposta explícita "score determinístico é inegociável").
- **Agente único com tool-calling livre (sem grafo de estados)**: mais flexível, porém perde a garantia de gates de negócio determinísticos (ex.: poderia chamar "agendar visita" antes de qualificar o lead) e aumenta custo/latência por turno; fica registrado como possível evolução pós-POC, não adotado agora.

---

## ADR-012: Catálogo de Imóveis Persistido em DynamoDB (substitui o backend de armazenamento da ADR-003)

**Context**
A ADR-003 especificou S3 como armazenamento do catálogo de imóveis, com o índice FAISS carregado em memória a partir desse objeto. Na prática, o upload do `properties.json` para S3 (`infra/s3.tf`, `aws_s3_object.properties_catalog`) foi provisionado, mas nenhum código da aplicação chegou a ler do S3 — o catálogo era carregado exclusivamente do arquivo JSON local empacotado com a Lambda/task (`apps/conversation-router/data/properties.json`, gerado por `scripts/seed_properties.py`). Isso deixava o armazenamento persistente do catálogo sem implementação real, divergindo do que a ADR-003 documentava. Solicitação explícita do time: migrar o catálogo para DynamoDB, reaproveitando o padrão já usado pelas outras 5 tabelas `sdr-*` (sessões, PII, alertas de restrição), mantendo o JSON gerado como fonte de seed.

**Decision**
Nova tabela `sdr-properties` (DynamoDB, `PAY_PER_REQUEST`, hash key `id`, sem GSI). `apps/conversation-router/service/properties_catalog.py` passa a carregar o catálogo padrão (cold start, sem `catalog=` explícito) via `Scan` paginado nessa tabela, com fallback fail-open para o JSON local (`_load()`, inalterado) caso a tabela esteja vazia, sem client configurado, ou a chamada falhe. O `start.sh` popula a tabela após o deploy (fase 5), a partir do mesmo `properties.json` gerado na fase [2/6], via `scripts/load_properties_dynamodb.py` (upsert idempotente por `id`, `batch_write_item` em lotes de 25). O `aws_s3_object.properties_catalog` (`infra/s3.tf`) permanece como está — não é removido nesta mudança. **O mecanismo de busca (índice FAISS/TF-IDF em memória, construído a partir do catálogo carregado) não muda** — esta ADR substitui apenas o backend de armazenamento persistente que a ADR-003 documentava, não a arquitetura de busca que ela também estabelece.

**Consequences**
**Positivos:**
- Reaproveita o padrão IAM/tabela já usado pelas outras 4 tabelas `sdr-*` — a policy wildcard `arn:aws:dynamodb:*:*:table/sdr-*` já cobre `sdr-properties` sem mudança de IAM.
- Elimina a divergência entre o que a ADR-003 documentava (S3) e o que o código de fato fazia (JSON local embutido, sem armazenamento persistente real).
- Reseed idempotente e não-destrutivo: rodar `start.sh` repetidamente apenas sobrescreve por `id`, sem duplicar itens.

**Negativos:**
- `Scan` completo da tabela no cold start em vez de um único `GetObject` do S3 — latência extra provavelmente desprezível no volume atual (~120-200 imóveis).
- `aws_s3_object.properties_catalog` permanece como artefato sem leitor na aplicação (decisão explícita de não remover nesta mudança).

**Alternatives Rejected**
- **Implementar a leitura do S3 conforme a ADR-003 original**: rejeitado por pedido explícito do time de usar DynamoDB, e por já existir o padrão de acesso/IAM/injeção de client DynamoDB replicável de `SessionStore`/`KmsPiiRegistry`/`DynamoRestrictionCheck`.
- **Remover `aws_s3_object.properties_catalog`**: avaliado e rejeitado por decisão explícita do time nesta rodada — mantido como está, sem remoção.

---

## ADR-014: Autenticação Cognito do Dashboard — Authorizer Conectado + Login Direto (USER_PASSWORD_AUTH), não Hosted UI

**Context**
A unidade u7-dashboard (code-generation) já havia deixado explícito em seu `code-summary.md` que a validação real do JWT ficaria para a "fase de infra": o handler de `GET /api/kpis` só checava a presença de um Bearer, e o login na UI era um placeholder por env (`DASHBOARD_API_TOKEN`). A fase de infra (Terraform) de fato provisionou o User Pool, o App Client e um `aws_apigatewayv2_authorizer` JWT (`infra/cognito.tf`), mas nunca completou a ligação: a rota `GET /api/{proxy+}` (`infra/apigateway.tf`) não referenciava esse authorizer, então a validação nunca rodava — qualquer string como Bearer passava pelo handler. Além disso, a ADR-005 (Dashboard como Streamlit Community Cloud) nunca foi seguida na prática: o dashboard roda em ECS Fargate (decisão de infra/deployment, IP público efêmero, sem ALB/domínio por custo), o que invalida o fluxo de Hosted UI/OIDC (`st.login()`) desenhado no PRD §10.3 — esse fluxo depende de uma `callback_url` estável, que não existe sem domínio fixo.

**Decision**
1. Rota `dashboard_proxy` (`infra/apigateway.tf`) passa a declarar `authorization_type = "JWT"` e `authorizer_id = aws_apigatewayv2_authorizer.cognito.id` — a validação de assinatura/issuer/audience/expiração passa a ocorrer na borda do API Gateway, antes de a Lambda `dashboard-api` ser invocada.
2. Login do dashboard usa `cognito-idp:InitiateAuth` (fluxo `USER_PASSWORD_AUTH`) direto do App Client, com formulário usuário/senha nativo do Streamlit (`apps/dashboard-ui/app.py`: `cognito_login`, `cognito_respond_new_password`), em vez de Hosted UI/OIDC. Trata o desafio `NEW_PASSWORD_REQUIRED` (todo usuário criado via `admin-create-user` nasce em `FORCE_CHANGE_PASSWORD`). `IdToken` fica só em `st.session_state` (memória da sessão, nunca persistido).
3. Nova IAM role de task `sdr-dashboard-ui-task` (`infra/ecs.tf`), restrita a `cognito-idp:InitiateAuth`/`RespondToAuthChallenge` no ARN do User Pool. Env vars `COGNITO_USER_POOL_ID`/`COGNITO_CLIENT_ID`/`AWS_REGION` injetadas no task definition; `DASHBOARD_API_TOKEN` removido (morto — o Bearer agora é o JWT real por usuário).
4. Outputs `cognito_user_pool_id`/`cognito_app_client_id` (`infra/cognito.tf`) para o passo operacional de criar as contas do time (`aws cognito-idp admin-create-user`) após o `terraform apply` — documentado no PRD §11 item 7.

**Consequences**
**Positivos:**
- Fecha o gap real de segurança: `/api/kpis` deixa de aceitar qualquer string como autenticação.
- Não exige ALB/domínio (custo adicional), compatível com a decisão de custo zero/mínimo já tomada para a POC.
- Reaproveita 100% dos recursos Cognito já provisionados (User Pool, App Client, Authorizer) — nenhum recurso novo de Cognito, só a ligação que faltava.

**Negativos:**
- Diverge do sketch original do PRD §10.3 (`st.login()`/Hosted UI) — mantido como esboço histórico no PRD, com nota de divergência apontando para este ADR.
- `USER_PASSWORD_AUTH` expõe a senha ao app (via `InitiateAuth`), em vez do redirect da Hosted UI nunca tocar a senha — aceitável para ~5-10 usuários internos do time, não para usuários externos.
- Sem refresh automático de sessão: expirando o `IdToken`, o usuário precisa logar de novo (sem fluxo de `REFRESH_TOKEN_AUTH` implementado nesta rodada).

**Alternatives Rejected**
- **Hosted UI/OIDC completo (`st.login()`, conforme o sketch do PRD)**: rejeitado nesta rodada por exigir domínio fixo + ALB (custo fora do orçamento da POC) para uma `callback_url` estável — o dashboard roda em ECS com IP público efêmero.
- **Resolver a ADR-005 (migrar de fato para Streamlit Community Cloud) para então usar Hosted UI**: fora de escopo desta rodada — tratado como drift pré-existente, não reaberto aqui.

---

## ADR-015: Interpretação de Linguagem Natural 100% pela LLM (com Roteador Ativo) e Filtro de Fotos no Crawler

**Context**
Conversas reais de teste (texto digitado e transcrição de áudio) expuseram que regex e palavras-chave ainda decidiam "entendi / não entendi" no caminho em que a LLM está ativa, desfazendo ou corrompendo o que ela interpretou: `extract_lead_structure` gravava como região qualquer palavra após "em/na/no" (`em qualquer momento`, `no terreno`, `na garagem`) e não entendia `mil metros quadrados`; `_OPTIONS_REQUEST_RE` cancelava o `request_human` escolhido pela LLM se a mensagem contivesse "opção"/"lista"/"tudo"; `visit_interest` só valia com uma lista fixa de frases; uma regex fixa tratava "pra quem" nas perguntas de reserva; o casamento de região comparava tokens ("São Caetano" casava com "São João Clímaco"). Em paralelo, o CMS da Gonçalves anexa aos anúncios uma colagem das fachadas da própria imobiliária (215 fotos em ~9 formatos) e um ícone cinza de "sem foto" — o bot enviava isso como "foto do imóvel".

**Decision**
1. Com `llm_router` ativo, **só a LLM interpreta** `lead_info` (metragem, bairro/cidade, orçamento, prazo, pessoas), favorito/rejeitado, intenção de visita e pedido de humano. `extract_lead_structure` e `_apply_commercial_memory` (regex) rodam apenas sem roteador ou quando a chamada falha (plano B da ADR-011).
2. O roteador recebe o vocabulário do catálogo (`known_places()`: cidade → bairros) e grava o nome exato; `_region_matches` compara texto normalizado (sem acento/pontuação), sem stopwords.
3. `visit_interest` exige `visit_quote` (trecho literal da mensagem); `validation.py` só verifica que a citação existe na mensagem. Remove `_VISIT_EVIDENCE_RE`.
4. `execute_tool("request_human")` não consulta mais palavra-chave da mensagem; removida a ramificação "pra quem/quem reservou" — a regra de nunca inventar quem reservou fica no prompt de resposta.
5. Para 1–2 imóveis em foco, `generate_reply` recebe a ficha completa (campos não vazios/não zero, sem URLs/ids) + descrição; o prompt manda tratar campo ausente como desconhecido e confirmar com o corretor.
6. Resposta repetida (≈ mensagem anterior do bot) é descartada em código e vale o texto oficial; reescrita do turno de pedido de contato precisa citar um canal (telefone/WhatsApp/e-mail) — sanidade da **saída**, sem restringir o modo de pedir.
7. Fotos genéricas são removidas **no crawler** (`drop_generic_images.py`: correlação da miniatura 32×32 sem margens brancas com `generic_image_signature.json`, limiar 0,65 numa lacuna medida: colagem ≥ 0,73 × foto real mais parecida 0,52; 404/410 removidas; erro de rede mantém). O bot não tem lógica de foto genérica. `start.sh` usa sempre o catálogo do crawler limpo; sem crawler, catálogo sintético (sem fotos).

8. **Correção pós-teste (privacidade):** `PII_PATTERNS["TELEFONE"]` ganhou alternativas para celular com o 9 separado por qualquer combinação de espaço/hífen/ponto e DDD com zero (`(11) 9-7991-8262`, `+55 11 9 7991-8262`, `011 9 7991 8262`). Antes, esses formatos passavam em texto puro ao provedor de LLM, não eram salvos no `pii_store` (o lead chegava ao CRM sem telefone) e o `check_output_leak` também não os reconhecia. A mascaragem segue 100% determinística: o telefone nunca depende de interpretação da LLM.
9. `property_detail` aceita `arguments.property_refs` (até 3 números de itens já exibidos) para pedidos como "fotos desses três"; o código resolve cada referência contra a lista e envia 1 foto de cada. `generate_reply` recebe, no turno de pedido de contato, a informação de que nada foi encaminhado ainda.

10. **Máscara de PII restrita a canais de contato:** `SecurityLayer.mask` mascara apenas e-mail, telefone (dígitos e por extenso) e CNPJ. Removido o padrão de NOME por capitalização e a lista de primeiros nomes. Motivo: falsos positivos em lugares/empreendimentos e no texto colado do bot bloqueavam respostas legítimas e impediam o fechamento do lead. `check_output_leak` mantém o bloqueio de e-mail/telefone/CNPJ e do nome do perfil do Telegram gravado no `pii_store`; se a reescrita da LLM for barrada, o handler usa `official_response` (texto determinístico do turno) antes do fallback genérico.

**Consequences**
**Positivos:** entende transcrição/fala solta/abreviações sem manutenção de regex; erros de interpretação passam a ser corrigidos no prompt (e cobertos pelo quality gate), não por mais remendos em Python; o bot passa a dizer a verdade quando não há foto.
**Negativos:**
- Custo fixo de ~500 tokens por turno no roteador (vocabulário do catálogo) e dependência da qualidade do modelo — mitigada pelo quality gate com LLM real (retry 1x só dos que falharem) no deploy.
- Sem LLM, a interpretação cai para regex simples (degradado, ciente).
- 32 imóveis (13% do catálogo) ficam sem foto até a imobiliária subir uma foto real.

**Alternatives Rejected**
- **Ampliar regex/dicionários** (mais ordinais, mais frases de visita, mais sinônimos de lugar): nunca converge; cada fala nova exigiria novo remendo.
- **Filtrar a foto no bot** (lista de hashes/URLs): a assinatura é específica do CMS de cada imobiliária e pertence à fonte de dados.
- **Reagir só ao `urlfotoprincipal`/nome de arquivo** (`-z-1-`): o mesmo conteúdo chega em vários nomes/formatos; só o conteúdo discrimina.

## ADR-016: Dashboard com Leads, Sessão Persistente, Sem Widget de Custo; CRM Real via HubSpot MCP

**Context**
O dashboard só mostrava contagens: o backoffice não conseguia falar com o lead. O login se perdia a cada F5 (`st.session_state`). O widget de custo LLM lia métricas CloudWatch que nenhum componente emite (sempre R$ 0,00). O CRM da POC (CSV em `/tmp` da Lambda) é efêmero e ilegível pelo time; o PRD (§8.9) prevê HubSpot via MCP.

**Decision**
1. **Leads no dashboard**: `GET /api/leads` (dashboard-api) lista perfil + conversa mais recente e decifra o contato no registro `sdr-pii` com KMS no momento da leitura; só usuário autenticado no Cognito acessa. Contato completo (sem máscara) porque o objetivo é o backoffice contatar o lead.
2. **Botão "Enviar ao HubSpot"**: `POST /api/leads/{id}/crm` publica na MESMA fila do handoff (Contrato 4); o `crm-adapter` trata igual à transição automática (mesma idempotência/monotonia de estágio).
3. **Sessão persistente**: cookie `sdr_session` com id aleatório; o refresh token do Cognito fica no servidor do dashboard (cofre em memória, TTL 12 h). Roubo do cookie não expõe token do Cognito. Reiniciar o container exige novo login.
4. **Custo LLM removido** do dashboard e das métricas exibidas.
5. **CRM real = HubSpot via MCP remoto** (`https://mcp.hubspot.com`, OAuth 2.1 + PKCE, refresh token de uso único). O MCP NÃO aceita token de app privado. Credenciais (`HUBSPOT_MCP_CLIENT_ID/_SECRET/_REFRESH_TOKEN`) entram por `secrets.local.env` → Secrets Manager. **Status: implementado e validado contra o HubSpot real.** `crm-adapter` usa `HubSpotCrmGateway` + `HubSpotMcpClient` (SDK `mcp`, streamable HTTP) quando `HUBSPOT_SECRET_ID` está definido; senão o CSV. Lead = contato (`firstname/lastname/email/phone`), estágio → `hs_lead_status`, qualificação → campo padrão `message` (sem propriedades custom). Tools usadas: `search_crm_objects` (busca por e-mail, depois telefone) e `manage_crm_objects` (create/update). Credenciais OAuth e refresh token (uso único, regravado a cada renovação) ficam na secret `sdr/hubspot-mcp`; `stop.sh` salva o token vigente em `secrets.local.env`. Limitação: o estágio conhecido fica em cache da instância Lambda, então a monotonia de estágio não enxerga mudanças feitas manualmente no HubSpot.

**Alternatives Rejected**
- Tabela de leads com contato mascarado: o backoffice não conseguiria contatar o lead.
- API REST do HubSpot atrás da interface MCP: contraria a decisão de usar MCP.
- Token de app privado para o MCP: não suportado pelo servidor remoto.
- Cookie com o próprio refresh token: exporia credencial do Cognito no navegador.

## ADR-017: Consentimento LGPD Só com Aceite Explícito; Ajustes de Experiência da Conversa

**Context**
Numa conversa real o bot pediu consentimento ("Podemos continuar?"), o lead respondeu "compra" e o fluxo gravou `consent_recorded=True` — qualquer resposta que não fosse "não" valia como aceite. Na mesma conversa o bot prometeu fotos que não enviou, narrou falha interna ("desencontro nos filtros"), demorou para dizer que o catálogo não tinha o tipo pedido (sala comercial) e mostrou `**negrito**` literal no Telegram.

**Decision**
1. **Consentimento explícito e determinístico (sem LLM)**: só aceite que COMEÇA com uma forma clara de "sim" (`sim`, `ok`, `pode`, `claro`, `aceito`, `concordo`...) grava consentimento. Recusa ("não") segue encerrando. Qualquer outra resposta reapresenta o pedido (`CONSENT_REASK_MESSAGE`, texto fixo, nunca reescrito pela LLM) e mantém o estado `elicitation`, sem coletar dados.
2. **Trava de promessa de fotos**: se a reescrita da LLM afirma estar enviando fotos e nenhuma vai no turno, vale o texto oficial (mesmo padrão da trava do pedido de contato). Limite de fotos por lista subiu de 3 para 4 imóveis.
3. **Texto puro**: markdown (`**negrito**`, `#`) é removido da resposta humanizada (o envio ao Telegram não usa `parse_mode`).
4. **Eco do contato**: ao chegar telefone/e-mail novo, o handler anexa "Anotei seu telefone (11) 97991-8262…" DEPOIS do check de vazamento, em código — o número nunca passa pela LLM.
5. **Prompt de humanização** (regras 17–19): não narrar mecânica interna; dizer logo, em uma frase, quando o tipo de imóvel do catálogo difere do pedido; variar aberturas e não comentar o tom do lead.

**Alternatives Rejected**
- Classificar o aceite por LLM: consentimento é decisão jurídica, precisa ser previsível e auditável.
- Tratar qualquer mensagem como aceite implícito: não é consentimento livre e informado.
- Filtrar por tipo de imóvel na busca: fora do escopo desta rodada (o `lead_info` não tem `property_type`); tratado só na comunicação.

## ADR-018: Identidade Visual do Dashboard (Logo W Levitt + Tema Azul-Marinho/Dourado)

**Context**
O dashboard usava o tema padrão do Streamlit, sem relação com a marca. O logo da W Levitt tem paleta própria (azul-marinho e dourado).

**Decision**
1. **Tema nativo** em `apps/dashboard-ui/.streamlit/config.toml` (`base=dark`, primária `#D8A858`, fundo `#0B1D3A`, superfície `#13294F`, texto `#F4EBD6`), extraído por amostragem de pixels do logo. Sem CSS injetado/`unsafe_allow_html`.
2. **Logo** em `apps/dashboard-ui/assets/logo.png` (cópia idêntica, byte a byte, do `Designer.png` da raiz do repo — o logo oficial, ícone de prédio + balão de chat; um teste compara o hash das duas), exibido no cabeçalho ao lado do título; ausência do arquivo não quebra a tela.
2b. **Favicon da aba** = o mesmo logo (miniatura de 128 px gerada em memória por `page_icon()`, sem alterar o arquivo), via `st.set_page_config(page_icon=...)`; sem isso o Streamlit mostra o ícone padrão dele. Sem o arquivo, cai no emoji 🏢.
3. **Gráficos** usam `BRAND_GOLD` (mesmo valor do `primaryColor`); teste garante que tema e constante não divergem.
4. `Dockerfile` copia `assets/` e `.streamlit/` (sem isso o container sobe com o tema padrão).

**Alternatives Rejected**
- CSS customizado via `st.markdown(unsafe_allow_html=True)`: frágil entre versões do Streamlit e amplia superfície de injeção.
- Reduzir/recomprimir o logo: o arquivo oficial é usado como está (1254×1254, ~1,8 MB) para não haver versão divergente; o Streamlit o exibe a 64 px.

## ADR-019: Gate de Qualidade Ponta a Ponta das Anomalias (Chat com LLM Real → Detector → Dashboard)

**Context**
O detector (FR9) tinha 87 testes unitários/integração, mas nenhum cobria o caminho completo: conversa real no bot → alerta gravado → alerta visível no dashboard. Simular anomalia pelo Telegram também era incerto: a nota (0–1) soma volume de mensagens (0,3), tamanho médio (0,2), palavras negativas (0,3) e horário fora de 08–19h (0,2); as mensagens do bot entram na média e diluem o resultado, e o limite de produção é 0,7.

**Decision**
1. Novo gate com LLM real em `apps/conversation-router/tests/quality/test_anomaly_e2e_quality.py`, executado pelo mesmo passo de qualidade do `start.sh` (pula sem chave de LLM). O chat roda no `ConversationRouter` com a fiação de produção e relógio fixo; as conversas gravadas alimentam o handler REAL do `anomaly-detector` e depois o `dashboard-api` (`GET /api/kpis`), cada um em subprocesso (módulos `handler/service/infra` colidem entre apps).
2. Casos: (a) chat noturno com 4 mensagens negativas de 3500 caracteres gera alerta, restringe o agendamento (FR9.4) e aparece no dashboard sem `features` (FR7.4); (b) chat comercial normal de dia não gera alerta (sem falso positivo).
3. O teste usa limite 0,4 (parâmetro `ANOMALY_THRESHOLD`); o limite de produção (0,7) segue inalterado — com ele a demo exigiria ~20 mensagens enormes à noite, e de dia nunca dispara (máximo 0,8 sem o fator horário, na prática bem menos).

**Alternatives Rejected**
- Baixar o limite de produção só para demonstrar: decisão de produto, não de teste.
- Mocar o LLM: o objetivo é exercitar a conversa real, incluindo o tamanho das respostas do bot que entra na média.

## ADR-020: Orçamento com Piso e Teto Interpretados pela LLM; Trava de Promessa de Trabalho Futuro; CRM Só com Texto

**Context**
Conversa real: "São Bernardo a partir de 1000 reais" virou "orçamento até mil reais" (piso lido como teto). Na mesma conversa o bot disse "Vou buscar apartamentos... Um momento!" quando a busca já tinha rodado no turno; o lead perguntou "cade?" e o bot pediu desculpa pela demora. Investigando, os logs mostraram a LLM devolvendo `budget` como número (1000) e o `crm-adapter` só aceita texto nesse campo — o lead seria descartado na fila (`message_rejected`, sem retry).

**Decision**
1. **Piso e teto vêm da LLM**, sem regra no código: o roteador extrai `budget` (teto: "até", "no máximo", "tenho X") e `budget_min` (piso: "a partir de", "acima de", "pra cima"); faixa preenche os dois; piso sem teto não copia o piso para `budget`. O código só aplica: busca exclui preço abaixo de 80% do piso (tolerância simétrica ao teto ×1,25) com o mesmo fail-open do teto, e o score favorece quem cabe na faixa; o qualifier usa `budget` ou, na falta, `budget_min`.
2. **Trava de promessa de trabalho futuro** (mesmo padrão da trava de fotos): se a reescrita promete "um momento", "vou buscar", "já volto", "te aviso quando surgir" ou "crio um alerta" e o texto oficial não promete, vale o texto oficial. "Vou confirmar com o corretor" continua permitido. O prompt de humanização ganhou a regra 19 (não há busca em segundo plano nem alertas).
3. **CRM só recebe texto**: `budget`, `deadline` e `area` passam por `_crm_text` antes de ir à fila (`budget_min` sem teto vai como "a partir de X").

**Alternatives Rejected**
- Regex/lista de expressões ("a partir de", "acima de") para separar piso de teto: nunca converge e contraria o ADR-015.
- Guardar só um valor e perguntar "é mínimo ou máximo?": a LLM entende a frase; perguntar de novo piora a experiência.
- Alterar o contrato do crm-adapter para aceitar números: o contrato (Contrato 4) é anterior e tem outros produtores/testes; converter na origem é mais simples.

## ADR-021: Tracing Distribuído com AWS X-Ray (Lambdas e conversation-router)

**Context**
O NFR5.2 (traços distribuídos) estava "Not Met" no `build-and-test`, e o NFR1.1 (primeira resposta < 10 s) não tinha como ser decomposto: os logs mostram 8–15 s por turno, mas não onde o tempo vai (LLM, DynamoDB, fallback de modelo).

**Decision**
1. **Lambdas**: `tracing_mode = "Active"` e `attach_tracing_policy = true` nos 7 módulos de Lambda. Nas 5 ativas (`crm-adapter`, `contact-ingest`, `anomaly-detector`, `followup`, `dashboard-api`) o `tracing.py` chama `patch(("botocore",))` do `aws-xray-sdk` no import: DynamoDB, SQS, KMS e Secrets Manager viram subsegmentos, e o SQS propaga o identificador do trace à Lambda consumidora. Contexto ausente só registra o erro (`AWS_XRAY_CONTEXT_MISSING=LOG_ERROR`).
2. **conversation-router (ECS)**: o ECS não abre segmento sozinho, então `server.py` cria um segmento por requisição (anotações `route` e `status`) e cada chamada à LLM é um subsegmento `llm:<modelo>` (anotações `model`, `max_tokens`; falhas ficam como `fault` com a exceção; `llm_fallback=true` marca o uso do Tier 2). O daemon roda como contêiner auxiliar da mesma task (`public.ecr.aws/xray/aws-xray-daemon:3.7.0`, UDP 2000 em 127.0.0.1), e a política `sdr_lambda` ganhou as ações `xray:*` necessárias.
3. **Sem SDK instalado** (testes locais) tudo vira no-op; observabilidade nunca derruba o atendimento.
4. **Segurança dos traces — só `botocore`, nunca `patch_all()`**: o `patch_all()` também instrumenta `requests` e `urllib` e grava a URL das chamadas de saída no trace. Como o `followup` e o router chamam `https://api.telegram.org/bot<TOKEN>/...`, o **token do bot iria para o X-Ray**; isso foi reproduzido com o SDK real e um daemon UDP simulado (token presente com `patch_all`, ausente com só `botocore`). O botocore registra operação, tabela e fila, não o conteúdo das mensagens, e as anotações do router são `route`, `status`, `model`, `max_tokens` e `llm_fallback` (sem texto de conversa nem PII). Cada `test_tracing.py` falha se alguém voltar a chamar `patch_all()`.
5. **Insumo do estágio `observability-setup`** (Operation, ainda não executado): os subsegmentos `llm:*` e a anotação `llm_fallback` são a evidência esperada para decompor o NFR1.1; a amostragem do X-Ray pode ser ajustada para o `anomaly-detector`.

**Limites conhecidos**
- O API Gateway **HTTP (v2)** não suporta X-Ray: o trace do webhook do Telegram começa no router, e o do dashboard começa na Lambda `dashboard-api`.
- O HTTP de saída (Telegram, OpenRouter, HubSpot) **não** aparece como subsegmento, de propósito (item 4); a latência da LLM é coberta pelos subsegmentos `llm:*`.
- `voice-adapter` e `dashboard-ui` (ECS) ficaram de fora: o valor está no router, onde está a latência da LLM.
- O `anomaly-detector` (a cada minuto) gera ~43 mil traces por mês, quase metade do plano gratuito (100 mil); a amostragem padrão do X-Ray já limita isso, e a regra pode ser ajustada.
- **Não validado na AWS**: o SDK foi testado com daemon UDP simulado e os testes automatizados; o daemon como sidecar do Fargate e o tracing nas Lambdas só se confirmam num deploy.

**Alternatives Rejected**
- OpenTelemetry/ADOT: mais flexível, mas exige coletor e configuração maiores para o mesmo ganho na POC.
- Tracing manual por logs de latência: não dá a visão encadeada entre serviços.
- `patch_all()` (instrumentação automática de HTTP): vazaria o token do bot (item 4).

## ADR-022: Nome da Assistente (Cecília) em SSM Parameter Store e Apresentação na Primeira Mensagem

**Context**
O bot se apresentava como "o assistente SDR da W Levitt", sem nome. Pediu-se uma persona com nome próprio, configurável sem novo deploy.

**Decision**
1. O nome fica no **SSM Parameter Store** (`/sdr/bot-name`, tipo `String`, valor padrão `Cecília` pela variável Terraform `bot_name`), no mesmo padrão dos parâmetros de modelo de LLM (ADR-010). O router recebe o nome do parâmetro em `BOT_NAME_SSM`; a role da task já lê `/sdr/*`, sem mudança de IAM.
2. `service/bot_identity.py::get_bot_name()` resolve o nome: **SSM → variável `BOT_NAME` → "Cecília"**, com cache de 5 minutos (trocar no SSM vale em poucos minutos, sem deploy), higieniza espaços e quebras de linha e limita a 40 caracteres (o nome entra no prompt da LLM).
3. A primeira mensagem passa a ser `consent_message(nome)`: "Olá! Meu nome é <nome> e sou assistente virtual da W Levitt, com foco em espaços corporativos. Para continuar, preciso do seu consentimento LGPD…". Redação neutra quanto a gênero, para o nome poder mudar. O texto continua **fixo e determinístico** (LGPD): `is_consent_message()` o reconhece com qualquer nome e a humanização por LLM nunca o reescreve.
4. O prompt de humanização recebe `SEU NOME: <nome>` (só quando há nome): ela não se reapresenta a cada resposta e só cita o nome se o lead perguntar quem ela é.

**Consequences**
- Um novo `terraform apply` volta ao valor da variável `bot_name` (`overwrite = true`, como nos modelos); para uma troca permanente, mudar a variável.
- O nome de exibição do bot no Telegram (`@RM369853_bot`) é do BotFather e **não** muda por aqui.
- Verificado só por testes automatizados; o parâmetro em si e a leitura pela task só se confirmam numa subida na AWS.

**Alternatives Rejected**
- Nome fixo no código: troca exigiria deploy.
- Variável de ambiente na task: também exige novo apply/restart, e o pedido era SSM.

## ADR-023: Rótulos em Português no Dashboard e no Resumo do HubSpot

**Context**
O dashboard mostrava códigos internos em inglês (`handoff`, `conversation`, `rent`, `high`, `alert_issued`, `atypical_hours`), o que confundia quem opera: "handoff" não diz que o lead foi passado a um corretor.

**Decision**
1. A tradução é só de **apresentação**: a API, o DynamoDB e as métricas continuam com os códigos estáveis em inglês. O `dashboard-ui` traduz estados da conversa (ex.: `handoff` → "Encaminhado ao corretor", `conversation` → "Em conversa", `greeting` → "Saudação", `elicitation` → "Consentimento"), intenção (compra, locação, investimento), urgência (alta, média, baixa) e os alertas (tipo, situação e ação tomada).
2. Valor desconhecido aparece **como veio** (nunca some nem quebra a tela); no gráfico de intenções, códigos e nomes já traduzidos são somados no mesmo rótulo.
3. O resumo gravado pelo `crm-adapter` no campo `message` do contato do HubSpot também traduz urgência e intenção, porque quem o lê é o corretor. O status do contato no HubSpot (`hs_lead_status`) é traduzido pelo próprio HubSpot.

**Alternatives Rejected**
- Traduzir na API ou no banco: quebraria consumidores e métricas que dependem dos códigos.
- Biblioteca de i18n: excesso para uma tela e 6 dicionários pequenos.

## ADR-024: Índice `lead-index` na Tabela de Alertas e Guarda de Índices DynamoDB

**Context**
O X-Ray mostrou, no `anomaly-detector`, `ValidationException: The table does not have the specified index: lead-index`. O código (U5 `alert_store.py` e o checador de restrição do router `restriction.py`) consulta o GSI `lead-index` na tabela `sdr-alerts`, como o desenho previa (ADR-004, FR9.4), mas o Terraform só declarava esse índice em `sdr-sessions`. Efeitos em produção: (1) o job de anomalias abortava ao chegar no primeiro lead normal, então os leads seguintes não eram avaliados nem alertados; (2) o checador do router é *fail-open* por decisão (indisponibilidade não bloqueia o lead), então a restrição de agendamento (FR9.4) **nunca era aplicada**, sem aviso. Os testes não pegaram porque os DynamoDB falsos aceitam qualquer `IndexName` (o gate ponta a ponta da ADR-019 também usa o falso).

**Decision**
1. `sdr-alerts` ganha o atributo `lead_id` (S) e o GSI `lead-index` (`hash_key = lead_id`, projeção `ALL`), igual ao de `sdr-sessions`. Um `terraform apply` sobre a tabela existente cria o índice online (tabela on-demand), sem recriar nem apagar alertas.
2. Novo guarda `tests/infra/test_dynamodb_indexes.py`, executado pelo `start.sh` antes do build: todo `IndexName` usado no código precisa existir no `infra/dynamodb.tf` para a tabela certa, as chaves dos índices precisam estar declaradas como atributos, e um índice novo no código obriga a registrá-lo no teste. Sem a correção ele reprova apontando as duas consultas quebradas.

**Alternatives Rejected**
- Trocar a consulta por `Scan` com filtro: lê a tabela inteira a cada lead e a cada mensagem (o checador roda no fluxo do lead).
- Tornar o checador *fail-closed*: um erro de infraestrutura passaria a bloquear leads legítimos.

## ADR-025: Dados de Busca Vindos de `arguments` e Log da Decisão do Roteador (a "lista fiel" foi desfeita)

**Context**
Num chat real (09/10) o lead pediu "opções de São Paulo", recebeu 3 itens numerados, digitou "1" e o bot listou 9 imóveis. A investigação (LLM real, 8 repetições do turno "1": `property_detail` em 0 de 8) achou:
1. **Região perdida**: a LLM às vezes manda `region` só em `arguments` (ex.: `request_options` com `region="São Paulo"`) e deixa `lead_info` vazio; o código buscava só com o `lead_info` do estado, então a busca perdia o filtro e o estado nunca guardava a região. No "1" seguinte a LLM via o pedido de São Paulo ainda sem resposta e listava de novo.
2. **Lista vista ≠ lista guardada**: o estado guardava 9 imóveis e a humanização reescrevia 3, em outra ordem.
3. **Sem observabilidade**: o log guardava só a ferramenta escolhida.

**Decision (em vigor)**
1. `validate_router_output` promove ao `lead_info` os dados de busca (`region`, `area`, `budget`, `budget_min`, `deadline`, `people_count`, `decision_maker`, `intent`) que a LLM mandou só em `arguments`; um `lead_info` explícito prevalece. O prompt do roteador também exige esses dados SEMPRE em `lead_info`. Medido com a LLM real: o "1" virou `property_detail` em 16 de 16 (duas conversas × 8), contra 0 de 8 antes.
2. **Log `roteador:`** a cada turno: ferramenta (e a que a LLM pediu, se a validação a mudou), `arguments`, chaves do `lead_info` extraído, nº de imóveis exibidos e raciocínio (200 caracteres). O texto do lead já aparece, mascarado, no log `turno IN`.

**Revertido em 2026-10-10 — "manter a lista oficial" na humanização**
A decisão original incluía também: regra 21 no prompt de humanização (todos os itens, mesma ordem e numeração), limite do JSON de imóveis de 1500 para 4200 caracteres e uma trava em código que descartava a reescrita se a numeração da lista mudasse. **Foi desfeita a pedido do responsável**, que viu o fluxo piorar numa conversa real. A reversão é completa no código; o problema que ela tentava resolver (a lista que o lead vê ter menos itens, ou outra ordem, que a guardada no estado) **continua em aberto**. Observação honesta: a reversão sozinha não resolveu o sintoma relatado ("ele nem oferece locação, compra ou investimento"): a causa dele era outra (ADR-026).

**Alternatives Rejected**
- Regex que reconheça "o 1"/"o primeiro" no código: contraria o ADR-015.
- Guardar no estado só os itens que a humanização citou: a LLM decidiria qual é a verdade do sistema.

## ADR-026: Humanização Sem Imóveis Antigos nas Fases de Consentimento e de Pergunta de Intenção

**Context**
Conversa real (09/10, 22:53): depois de "oi" e "sim" o bot respondeu "Perfeito! Encontrei algumas opções interessantes para locação…" e listou imóveis, em vez de perguntar se o lead busca compra, locação ou investimento. A resposta OFICIAL do sistema era a pergunta de intenção, mas a camada de humanização recebia, no payload, os 9 imóveis que já estavam guardados no estado (o roteador roda também no turno do consentimento e deixa imóveis no contexto).

**Decision**
`SalesFlow._properties_for_reply(state)` define o que a humanização pode citar: os imóveis do turno (`response_properties`) sempre; os guardados de turnos anteriores (`properties`) só **depois** das fases `greeting`, `elicitation` e `intent`. Nessas fases a resposta oficial é uma pergunta ou o texto do consentimento, e nada foi recomendado ainda.

**Evidência (LLM real, cenário "oi" → "sim")**
Teste A/B, 8 execuções por variante: código com imóveis no payload, 1/8 respostas corretas e 7/8 inventando a intenção ou listando imóveis; **sem a linha de nome no prompt**, 1/8 (o nome da assistente não era a causa); **sem imóveis no payload**, 8/8. Com a correção aplicada: **12/12**.

**Consequences**
- Depois da fase de intenção nada muda: respostas sobre imóveis já mostrados ("e o terceiro?") continuam recebendo `properties`.
- Cobertura: `tests/unit/test_reply_properties.py` (8 testes). O gate de qualidade com LLM real continua passando.

**Alternatives Rejected**
- Reforçar o prompt ("não liste imóveis quando a resposta for uma pergunta"): depende de a LLM obedecer; remover o dado que a induz ao erro é determinístico.
- Não rodar o roteador no turno do consentimento: mexe no comportamento de extração do primeiro turno e não era necessário.

## ADR-027: Aceite do Consentimento Interpretado pela LLM (substitui a lista de palavras da ADR-017)

**Context**
A ADR-017 passou a exigir que a resposta ao pedido de consentimento começasse com uma palavra de uma lista ("sim", "ok", "pode", "claro"...). Isso evitou que "compra" valesse como aceite, mas deixou o bot com cara de URA: "tô de acordo", "manda ver", "fechou" ou um 👍 eram tratados como resposta não entendida e o pedido era repetido. O responsável pediu que a LLM entenda a resposta naturalmente, sempre lendo a conversa inteira.

**Decision**
1. `llm.classify_consent(message, conversation_history)`: uma chamada à LLM (Tier 1 com fallback, `temperature=0`, JSON) que lê o histórico (inclui o pedido de consentimento) e devolve `accepted`, `refused` ou `unclear`. O prompt manda responder `unclear` na dúvida: consentimento exige concordância clara; responder outro assunto ("compra") não é aceite.
2. `SalesFlow._consent_decision` usa essa decisão no nó `elicitation`: `accepted` grava o consentimento e segue; `refused` encerra; `unclear` refaz o pedido (texto fixo). O texto do pedido continua fixo e nunca é reescrito pela LLM.
3. Sem LLM configurada, ou se ela falhar ou devolver algo inválido, vale o **plano B** por palavras (o comportamento da ADR-017), no mesmo padrão do resto do fluxo (ADR-015).

**Evidência**
LLM real, 23 frases (13 aceites em formas variadas, 5 recusas, 5 "nem um nem outro"), duas rodadas: 23/23 nas duas. Gate de qualidade com LLM real: 58/58 (os 49 anteriores + 9 de consentimento). Testes: `tests/unit/test_consent_llm.py` (15) e `test_llm_reads_the_consent_answer_naturally` no gate.

**Consequences**
- Uma chamada a mais à LLM só no turno da resposta ao consentimento (~1–3 s).
- O registro do aceite depende da interpretação da LLM; a mitigação é a regra "na dúvida, unclear" e o teste permanente no gate.

**Alternatives Rejected**
- Ampliar a lista de palavras: nunca cobre a fala real e é o comportamento de URA que se quer evitar.
- Voltar a "qualquer coisa que não seja 'não' é aceite": aceite tácito não vale como consentimento LGPD.

## ADR-028: Decisão de Fechar Vai ao Corretor com o Imóvel Escolhido; o SDR Pega o Contato do Lead

**Context**
Teste real (10/10): "quero fechar o terceiro" virou `express_visit_interest`, cuja resposta oficial não pedia o contato, e a humanização ofereceu "passar o WhatsApp do corretor" ao lead. O SDR deve capturar o contato do lead, nunca entregar o do corretor. O imóvel escolhido também não aparecia no lead (dashboard/HubSpot), e a tabela de leads mostrava vazios campos que a conversa já conhecia (região, metragem, orçamento, prazo). A mensagem inicial estava longa (CNPJ, 90 dias, revogação).

**Decision**
1. **Roteador (prompt, interpretação da LLM)**: "DECISÃO DE FECHAR" — quando o lead quer fechar/comprar/alugar/ficar com um imóvel já exibido, usar `request_human` com o imóvel em `property_ref` e o título exato em `memory_updates.favorite_property`. E "ÚLTIMA MENSAGEM MANDA": o roteador decide pela mensagem do turno; o histórico é contexto já atendido (corrigiu o "1" respondido como se fosse o pedido anterior: 12/12, antes 9/10 a 12/16).
2. **Contato do lead**: `request_human` sem contato já caía no pedido de WhatsApp/e-mail (`_ask_contact`); o `express_visit_interest` sem contato passou a fazer o mesmo, com a ação pendente "falar com o corretor" retomada quando o contato chega. Prompt de humanização: NUNCA oferecer telefone, WhatsApp ou e-mail do corretor ou da imobiliária.
3. **Lead completo**: o handoff manda ao CRM também `region` e `property` (imóvel escolhido); o `crm-adapter` os aceita e o resumo do HubSpot mostra "Região" e "Imóvel escolhido". A tabela de leads (`/api/leads`) preenche intenção, orçamento (inclui "a partir de"), metragem, região e prazo com o que a conversa já sabe quando o perfil ainda não tem, e ganhou as colunas "Prazo" e "Imóvel escolhido". O perfil continua prevalecendo quando tem valor.
4. **Mensagem inicial curta**: "Para te atender, uso o seu nome do Telegram e vou te pedir só o seu WhatsApp ou e-mail, para um corretor falar com você (LGPD). Podemos continuar?" (sem CNPJ, 90 dias e revogação, a pedido do responsável; recusa e revogação continuam funcionando no fluxo).

**Evidência (LLM real)**
"quero fechar o terceiro" com lead sem contato: `request_human` + favorito correto 4/4 e teste de qualidade 3/3 (pede o WhatsApp/e-mail do lead, não oferece o do corretor, grava o imóvel). Gate completo: 59/59.

**Consequences**
- A mensagem inicial deixou de citar retenção (90 dias) e o direito de revogar; o comportamento (TTL, "não" encerra) não mudou.
- O CSV do CRM simulado não ganhou colunas (cabeçalho fixo); os campos novos vão ao HubSpot.

## ADR-029: Fatos de Fotos, Valor Nunca Vira IPTU e Sem Oferecer Financiamento

**Context**
1. Chat real (10/10): o imóvel tinha **1 foto** no catálogo (em 252 imóveis, 180 têm 9; esse tinha 1). O lead pediu "mais fotos" duas vezes. O bot não tinha o que enviar (comportamento certo), mas a resposta foi ruim: na 1ª vez a trava de promessa de fotos descartou a reescrita e devolveu a ficha seca, sem dizer que não havia mais fotos; na 2ª a LLM disse "envio agora mais fotos", uma promessa falsa que a trava não reconhecia ("envio agora" sem "te").
2. Subida do `start.sh` (10/10): o teste `test_detail_missing_from_the_sheet_is_not_invented` falhou numa rodada. Com a LLM real a falha ocorre em ~3–10% das vezes (3/30 no código atual, 1/30 no `HEAD` anterior; amostra pequena, o defeito já existia): ao perguntar "quanto é o IPTU?" de um imóvel sem IPTU na ficha, a LLM respondeu "IPTU de R$ 3 mil por mês", tomando o aluguel por IPTU.

**Decision**
1. **Fatos de fotos**: em detalhe de 1–2 imóveis o fluxo marca quantas fotos de cada imóvel já foram enviadas (`SalesFlow._with_photo_counts`, sem alterar o estado guardado) e o payload da humanização ganha `fotos_total` e `fotos_restantes` (nunca as URLs). A regra 10 do prompt diz: se o lead pede mais fotos e `fotos_restantes=0` e nada vai neste turno, dizer com naturalidade que essas são todas as fotos disponíveis, sem prometer mais. Listas de 3 ou mais imóveis não recebem esses campos (o payload delas já é limitado).
2. **Trava de promessa de fotos** passa a reconhecer "envio/mando agora/já/mais/as/essas/outras ... fotos" além das formas anteriores.
3. **Prompt**: na regra "nunca invente fatos", o preço ou aluguel nunca é IPTU, condomínio ou outra taxa; campo ausente de IPTU/condomínio/taxa vai ao corretor, sem citar valor.

**Evidência (LLM real)**
Fluxo completo reproduzindo o chat (imóvel com 1 foto já enviada, "quero mais fotos"), 12 execuções: 0 falsas promessas e 11/12 dizem que são as fotos disponíveis. Teste permanente: `test_more_photos_when_the_property_has_only_one_says_so_instead_of_promising` (8/8). A correção do IPTU é só prompt e **não foi medida** depois da mudança: o teste existente do gate (`test_detail_missing_from_the_sheet_is_not_invented`) é quem a valida no `start.sh`.

**Consequences**
- O texto de "essas são as fotos disponíveis" vem da LLM; se ela insistir em prometer, a trava reverte para a ficha seca (comportamento anterior).
- A taxa residual de oscilação do teste do IPTU deve cair, mas só o `start.sh` seguinte confirma.

**Adendo (2026-10-10) — financiamento e condições de pagamento**
Chat real: o bot perguntou "quer saber sobre condições de financiamento?" sem ter essa informação. Regra 21 do prompt de humanização: o bot não tem dados de financiamento, entrada, parcelas, taxas, FGTS, consórcio ou pagamento; **nunca oferece** o assunto; se o lead **perguntar**, diz em uma frase que o corretor explica, sem números, e segue o fluxo (pede o contato se ainda faltar). Medição (LLM real, 16 execuções): oferta espontânea 0/16 antes e depois (não reproduzi a oferta do chat); ao ser perguntado, remeter ao corretor passou de 14/16 para 16/16, sem inventar número. Teste permanente: `test_financing_is_never_offered_and_questions_go_to_the_broker` (5/5).

## ADR-030: Valor do Imóvel Escolhido em Coluna Própria (o Orçamento Fica Vazio se o Lead Não Informa)

**Context**
Um lead fechado apareceu no dashboard sem orçamento. Auditoria dos dados: o lead nunca informou valor (o `lead_info` só tinha intenção e região) e o bot não pergunta orçamento por conta própria; não era falha de gravação. Foi proposto preencher o orçamento com o preço do imóvel escolhido.

**Decision**
Não misturar os dois dados. Orçamento é o que o lead pode gastar (alimenta o score); o valor do imóvel é o que ele escolheu. O orçamento fica vazio quando o lead não informa, e o valor do imóvel escolhido ganha campo próprio, `property_price`: "R$ 850.000" ou "R$ 4.500/mês", formatado a partir do preço numérico do catálogo (o `price_text` é arredondado, ex.: "R$ 0.8 milhão") e achado entre os imóveis exibidos pelo título do favorito. Aparece na coluna "Valor do imóvel" do dashboard, na mensagem ao CRM e no resumo do contato no HubSpot ("Valor do imóvel"). Para leads antigos o valor é calculado na leitura, a partir da conversa.

**Não feito (decisão do responsável)**
Perguntar o orçamento ao lead de forma natural antes do handoff ficou de fora, por não ter sido pedido.

**Consequences**
- Corretor e dashboard veem os dois valores sem ambiguidade; o score não é distorcido.
- Sem preço numérico no catálogo, cai no `price_text`; sem favorito ou preço, fica vazio.
