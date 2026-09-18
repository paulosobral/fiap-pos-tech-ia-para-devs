**Collaborator:** aidlc-developer-agent

## Contribution

### Parecer de implementabilidade

As stories são tecnicamente realizáveis como POC, desde que o escopo fique limitado a dados sintéticos, calendário simulado, Telegram como canal único e integração HubSpot com adaptador simulado por padrão. O fluxo principal está bem decomposto: entrada, intenção, qualificação, recomendação, agendamento, handoff e observabilidade. A separação entre must-have e nice-to-have é adequada para prazo curto.

### Sizing estimado

Estimativas consideram uma pessoa familiarizada com Python/AWS e incluem implementação, testes básicos e integração local; não incluem espera por credenciais, validação de negócio ou troubleshooting de serviços externos.

- US1.1 — Iniciar conversa: 1 dia. Webhook Telegram, sessão DynamoDB, consentimento e idempotência.
- US1.2 — Texto livre: 1–2 dias. Contexto, TTL, persistência e botões inline.
- US1.3 — Voz: 2–3 dias. Download Telegram, conversão ffmpeg, empacotamento Lambda/container, faster-whisper, timeout e fallback. Maior risco de performance.
- US2.1 — Intenção: 1–2 dias. Classificação estruturada, limiar de confiança, persistência e conjunto de avaliação para comprovar 85%.
- US2.2 — Informações básicas: 1–2 dias. Máquina de estados/questionário adaptativo, deduplicação de respostas e validação de campos.
- US3.1 — Score: 1 dia. Regras determinísticas versionadas, score 0–100, urgência e testes de tabela.
- US3.2 — Handoff: 1–2 dias. Critério de prontidão, confirmação, bloqueio por anomalia e controle de estado.
- US4.1 — Buscar imóveis: 2–3 dias. Ingestão JSON/S3, embeddings/FAISS, filtros estruturados, carregamento na Lambda e fallback. Deve validar latência em cold start.
- US4.2 — Apresentar opções: 1 dia. Formatação Telegram, limite de três, escolha e pedido de mais resultados.
- US5.1 — Agendar visita: 1–2 dias. Calendário simulado, concorrência/idempotência, ICS, notificação e confirmação.
- US6.1 — Gerar handoff: 1–2 dias. Template Markdown, sanitização, criptografia, envio interno e garantia de completude.
- US7.1 — Dashboard: 2–3 dias. API de KPIs, Streamlit, Cognito/JWT, gráficos e tabela de anomalias. Não é realmente independente se os eventos ainda não existirem.
- US9.1 — Anomalias: 3–5 dias. Extração de features, pipeline Isolation Forest/PCA/Autoencoder, calibração, alertas, restrição de agendamento e falso-positivo documentado. Deve ser tratado como spike técnico + story de produto.
- US11.1 — HubSpot: 2–3 dias. SQS, Lambda adapter, retry/DLQ, modo simulado, REST Private App e demonstração MCP. Credenciais e formato Kanban podem ampliar prazo.
- US10.1 — Roleta: 1–2 dias. Regras configuráveis, lock/idempotência, rodízio, especialista por metragem e auditoria. Depende de cadastro/disponibilidade de corretores.
- US8.1 — Follow-up: 1–2 dias. EventBridge, cadência dia 2/5/9, janela de silêncio, cancelamento após resposta e memória.
- US12.1 — E-mail: 2–3 dias. SES, parsing, validação, deduplicação e abertura de sessão; depende de domínio/verificação e formato dos portais.

O total dos must-have é aproximadamente 22–35 dias-pessoa, antes de hardening, IaC, observabilidade, segurança, demonstração e correções. Para prazo de hackathon, recomenda-se entregar primeiro happy path demonstrável e mocks confiáveis; voice, anomalias e HubSpot devem ter critérios de fallback explícitos.

### Integrações e desenho técnico

- RAG: usar metadados estruturados para filtros exatos e FAISS apenas para similaridade semântica. O resultado do índice deve carregar identificador da oferta e retornar somente registros existentes; prompt deve receber contexto fechado e resposta deve validar IDs. Empacotamento do FAISS e modelo de embeddings em Lambda pode exceder tamanho/memória; avaliar Lambda container image, camada S3 ou serviço pré-carregado.
- Detecção de anomalias: executar job assíncrono diário fora do caminho conversacional. Persistir features, versão do modelo, threshold, score e explicação mínima do alerta. Isolation Forest/PCA/Autoencoder simultaneamente aumenta complexidade; definir regra de combinação e dataset de calibração antes de prometer 85% ou baixa taxa de falso-positivo.
- Roleta: implementar como serviço determinístico e transacional, com configuração versionada, chave de idempotência e registro de decisão. A regra de área não basta: definir disponibilidade, capacidade, especialidade, fallback e comportamento em empate.
- Telegram/voz: webhook deve responder rápido e delegar processamento pesado para fila. STT pode rodar em container/worker assíncrono; limitar duração/tamanho, aceitar timeout e informar processamento pendente. WhatsApp não deve entrar na POC; manter interface de canal agnóstica para roadmap.
- HubSpot: SQS→Lambda com DLQ é adequado. Outbox/evento de integração evita bloquear handoff. Definir mapeamento de propriedades, estágio Kanban, deduplicação por lead e política de reprocessamento.
- Calendário/ICS: calendário simulado precisa de contrato equivalente ao futuro provedor, timezone explícito (`America/Sao_Paulo`), idempotência e prevenção de dupla reserva.
- Segurança: mascarar PII antes de qualquer chamada ao LLM; separar identificador interno de dados criptografados; auditar consentimento, acesso, handoff e integrações. Evitar que logs estruturados contenham texto bruto ou tokens.

### Dependências entre stories

- US1.1 é fundação de US1.2 e US1.3.
- US1.2 → US2.1 → US2.2 → US3.1 → US3.2.
- US2.2 habilita US4.1 → US4.2 → US5.1.
- US3.2 habilita US6.1, US11.1 e US10.1; US8.1 também depende de estado e memória definidos no fluxo de qualificação.
- US7.1 deve consumir eventos/KPIs produzidos desde US1.1, US3.1, US5.1 e US9.1; portanto é paralelo em interface, mas não independente em dados.
- US9.1 depende de histórico suficiente ou dataset sintético calibrado e alimenta o bloqueio de US3.2/US5.1.
- US12.1 pode ser isolada, mas requer contrato de criação de sessão equivalente a US1.1.

### Riscos técnicos

- Meta de US1.3 (<15s) pode falhar em cold start, download, ffmpeg e modelo Whisper dentro de Lambda.
- Meta de intenção >=85% não é verificável sem corpus rotulado, definição de classes ambíguas e protocolo de avaliação.
- FAISS/modelos em Lambda podem causar cold start, limite de pacote e pressão de memória.
- Pipeline de três técnicas de anomalia pode gerar alertas inconsistentes, especialmente com poucos dados sintéticos.
- Telegram webhook, EventBridge, SQS e callbacks exigem idempotência para evitar mensagens, handoffs e reservas duplicadas.
- Cognito no Streamlit Community Cloud e APIs privadas exigem esclarecer arquitetura de autenticação, CORS, armazenamento de segredos e exposição da API.
- Critérios de LGPD, mascaramento e “desmascarar PII” no handoff precisam de fronteiras de acesso e auditoria testáveis.
- Custo estimado pode ser subestimado por STT, embeddings, egress, logs, invocações e chamadas LLM; estabelecer orçamento/limites e métricas.
- Stories declaradas como 1–3 dias não refletem cross-cutting de IaC, testes, observabilidade, segurança e deploy; reservar uma camada técnica transversal.

### O que falta definir

- Contrato de eventos e esquema DynamoDB: sessão, mensagens, lead, score, anomalia, handoff, agendamento e auditoria.
- Provedor/modelo de embeddings, estratégia de atualização do índice FAISS e comportamento quando filtros não retornarem resultados.
- Corpus e métrica de avaliação para intenção, score e anomalias; definir casos de baixa confiança e revisão humana.
- Regras completas da roleta, cadastro de corretores, disponibilidade e fallback.
- Política de consentimento, revogação, exclusão, retenção por tipo de dado e papéis autorizados a ver PII.
- SLA de processamento assíncrono e mensagens de fallback para voz, RAG, HubSpot e calendário.
- Contrato `GET /api/kpis`, origem dos agregados, janela temporal e timezone.
- Mapeamento HubSpot: propriedades obrigatórias, pipeline/stages, chave de deduplicação e comportamento em falha permanente.
- Estratégia de deploy de modelos/ffmpeg e limite operacional da POC; separar worker pesado de Lambda síncrona se necessário.
- Confirmação dos KPIs com gestor W Levitt e decisão final sobre quais nice-to-have entram na entrega.

### Pontos fortes e ajustes concretos

Pontos fortes: fluxo principal coerente, critérios mensuráveis, priorização MoSCoW, preocupação explícita com LGPD, DLQ, TTL, auditoria, RAG fechado e bloqueio por anomalia. Ajustes: separar stories de infraestrutura/cross-cutting das stories de negócio; transformar US9.1 em spike de calibração + implementação; corrigir a independência declarada de US7.1; adicionar critérios de idempotência, fallback, timezone, confiança e observabilidade; tratar voice como processamento assíncrono e não como requisito síncrono rígido; manter WhatsApp explicitamente fora da POC.

## Positions

- AGREE: Fluxo de atendimento, qualificação, RAG, agendamento, handoff e integração CRM está alinhado ao objetivo da POC.
- AGREE: Priorização Must Have versus Nice-to-Have é adequada, desde que o prazo reservado para integração e hardening seja contabilizado.
- AGREE: Uso de DynamoDB, SQS/DLQ, EventBridge, Lambda, S3, Telegram e Streamlit atende às restrições técnicas.
- OBJECT: US7.1 não é totalmente independente; dashboard depende de eventos, API de KPIs, autenticação e dados produzidos por outras stories.
- OBJECT: US9.1 é grande demais para 1–3 dias e mistura engenharia de features, treinamento/calibração, operação, UX de alerta e regra de bloqueio.
- OBJECT: US1.3 não deve garantir processamento síncrono <15s sem definir hardware, tamanho máximo, cold start e fallback assíncrono.
- OBJECT: US2.1 precisa de corpus rotulado e protocolo de medição para tornar o requisito de 85% verificável.
- OBJECT: US4.1 precisa definir estratégia de empacotamento/carregamento de FAISS e modelo, pois a restrição de Lambda pode inviabilizar a latência em cold start.
- OBJECT: US10.1 precisa de regras de disponibilidade e fallback além do corte de 500 m² para ser implementável sem decisões ocultas.
- AGREE: WhatsApp deve permanecer no roadmap; arquitetura pode expor contrato agnóstico de canal sem implementar integração agora.
