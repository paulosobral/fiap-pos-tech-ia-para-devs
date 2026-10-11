# Initiative Brief — Agente SDR Imobiliário B2B (W Levitt)

> Porta de aprovação da fase Ideation. Compila intent-statement, stakeholder-map, competitive-analysis, feasibility-assessment, constraint-register, scope-document, intent-backlog e wireframes. Conversa: pt-BR.

## Intent e problema

A W Levitt negocia imóveis corporativos/comerciais em São Paulo (lajes, andares, conjuntos, salas, terrenos/edifícios). O primeiro atendimento e a triagem de leads dependem de horário comercial e de trabalho manual: resposta inicial lenta, leads parados sem follow-up, corretor consumindo tempo com conversas brutas em vez de resumos qualificados, e detecção de intenção (compra/locação/investimento) dependente de processo manual.

O **Agente SDR B2B** resolve isso com atendimento conversacional automatizado 24×7, qualificação com pontuação, follow-up com contexto e handoff qualificado ao time de vendas — entregue como POC para o hackathon FIAP (Fase 5, PosTech IA para Devs).

## Validação de mercado

- **Nicho vazio**: Concorrentes diretos (Lais.ai, Maya/PLAZA, Squad) é especializado em imóveis corporativos/comerciais B2B.
- **Substituto dominante**: processo manual (planilhas + SDR/corretor) — exatamente a dor que o Agente SDR ataca.
- **Posicionamento**: "SDR de IA especializado em imóveis corporativos/comerciais B2B — qualificação proprietária com score explicável, detecção de anomalias e custo de operação fully serverless."
- **Objetivo**: diferencial competitivo do hackathon (demonstração/avaliação); go-to-market no roadmap pós-POC.

## Viabilidade e riscos

**Veredito: VIÁVEL.** Arquitetura 100% serverless, custo ~R$15/mês com modelo econômico (OpenRouter com Claude Haiku) + Secrets Manager. Compliance (LGPD) viável com POC 100% em dados sintéticos end-to-end.

Riscos principais e mitigação:
- **OpenRouter SPOF (ALTO)** → crédito pré-carregado, cap de tokens, fallback Bedrock documentado.
- **Cold start vs NF-03 <4s (MÉDIO)** → warm ping, imports lazy, medir p90.
- **Pacote Lambda 250MB (MÉDIO)** → pré-computar/treinar offline, artifacts no S3, layers separadas.
- **Deadline (risco de escopo)** → MoSCoW sequencial, núcleo primeiro, congelar infra ≥1 semana antes da gravação.

## Limite de escopo (POC)

**In-scope (must-have):** núcleo conversacional (Telegram → router → RAG → qualificação → handoff → dashboard); detecção de anomalias (IF+PCA+GLR+Autoencoder, job diário, alerta dashboard, restrição de agendamento para suspeitos); CRM HubSpot (MCP numa demo única + crm-adapter SQS→Lambda); voice (faster-whisper em Lambda layer). **Nice-to-have:** roleta + esteira Kanban.

**Out-of-scope (roadmap):** follow-up com contexto + calendário (ICS); ingestão de e-mail (SES, C7); WhatsApp; reporte formal periódico; auditoria formal imutável; Bedrock produção/sa-east-1; MCP-HubSpot embutido em produção.

**Limite rígido = data da gravação do vídeo** (não 12/10). Dados 100% sintéticos (imóveis + leads + CRM).

## Conceito (wireframes)

- **W1** — Dashboard visão geral: KPIs (leads 1.248, qualificados 62%, intenção 87%, 1ª resp. 4s), esteira Kanban (Novo/Qualif./Em negoc./Anomalia), timeline + alertas de anomalia.
- **W2** — Dashboard detalhe do lead: timeline (entrada→qualificação→rota→handoff→anomalia), ações (importar MCP HubSpot, agendar, atualizar status, reclassificar/manter suspeito).
- **W3** — Telegram: app existente, sem UI a desenvolver (só comportamento do bot).
- **W4** — MCP Inspector: demo única ao vivo (OAuth/PKCE), não é UI do produto.

## Plano de entrega

- **Dev solo** (time-formation pulado — sem team-assessment; capacidade Python+AWS confirmada).
- Sequenciamento dependency-first: (1) núcleo conversacional → (2) roleta/esteira → (3) anomalias → (4) CRM HubSpot → (5) voice.
- Congelar infra ≥1 semana antes da gravação; teardown `terraform destroy` pós-POC.

## Recomendação go/no-go

**GO.** Intent e escopo claros e aprovados pelo usuário; mercado valida o nicho B2B corporativo; viabilidade técnica e financeira confirmada; riscos com mitigação; POC dimensionada para o prazo. Recomendo aprovar a fase Ideation e avançar para Inception.

## Review

Artefatos de origem: intent-statement, stakeholder-map, competitive-analysis, feasibility-assessment, constraint-register, scope-document, intent-backlog, wireframes, user-flow. Achados do review de rough-mockups (R-01..R-05) são refinamentos a tratar em refined-mockups na fase Inception.