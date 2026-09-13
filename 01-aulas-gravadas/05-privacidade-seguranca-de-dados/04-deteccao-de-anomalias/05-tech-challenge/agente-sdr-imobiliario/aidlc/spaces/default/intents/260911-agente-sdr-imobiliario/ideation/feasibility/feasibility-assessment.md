# Feasibility Assessment — Agente SDR Imobiliário (W Levitt)

> Estágio Feasibility & Constraints (Ideation). Fonte: intent-statement, market-research (competitive-analysis, market-trends, build-vs-buy), feasibility-questions (Q1–Q6), pareceres dos agentes aws-platform e compliance.

## Veredito

**VIÁVEL.** Arquitetura 100% serverless bem desenhada (Lambda síncrona modular, sem Lambda→Lambda síncrono). Custo ~R$15/mês atingível **com duas correções de estimativa** (SSM no lugar de Secrets Manager; modelo LLM econômico no lugar de Claude Haiku pago a 5k msgs). Nenhum bloqueador AWS identificado. Compliance: viável, sem bloqueio, desde que a POC opere com dados sintéticos end-to-end e mantenha os controles LGPD do PRD.

> ⚠ Maior risco do projeto não é infra: é **escopo** (16 módulos lógicos) contra o deadline 12/10 (ideal antes, para gravação do vídeo).

## Viabilidade técnica por componente (AWS)

| Componente | Fit | Observações |
|---|---|---|
| Conversation orchestration (`conversation-router`) | Excelente | Monólito modular em Lambda; cold start 1–3s (pode estourar NF-03 <4s a frio → warm ping antes/durante demo); usar **HTTP API** (~3× mais barato) |
| LLM (OpenRouter Claude Haiku via LiteLLM) | OK, custo otimista | LiteLLM com `LLM_PROVIDER=openrouter\|bedrock` correto; R$8–15 só fecha com modelo econômico (DeepSeek V3 ~8× menor, bom PT-BR) + volume demo controlado; Haiku = upgrade documentado |
| RAG (`properties-rag`: FAISS memória + S3) | Excelente | ~200 docs cabe em RAM; **pré-computar embeddings offline** → S3 (evita PyTorch no pacote 250MB) |
| Lead-router (roleta) + esteira/Kanban | Total | Módulo interno; rotas em config (DynamoDB/SSM), não hardcode; timeline no DynamoDB |
| Detecção de anomalias (`anomaly-detector`) | OK c/ restrições | IF+PCA+GLR+Autoencoder; **treinar offline, Lambda só inferência** (artifacts joblib/npz no S3); dados sintéticos = história de demo, não detector de produção |
| Dashboard (Streamlit Community Cloud) | Adequado | Fora da AWS, custo zero; dorme por inatividade → abrir antes da gravação; fallback local; `st.login`+Cognito exige Streamlit ≥1.42 |
| CRM (`crm-adapter`: SQS→Lambda→HubSpot) | Bom | Separar MCP (demo única) do adapter (Private App REST) correto — OAuth/PKCE do MCP não funciona em Lambda assíncrona; token em SSM |
| Voice (faster-whisper Lambda layer) | Possível, frágil | Modelo base/tiny em S3 + download lazy para `/tmp`; ou Amazon Transcribe (~R$3/mês) |
| Ingestão e-mail (SES `contact-ingest`, C7) | Condicional | **SES sandbox** em conta nova — solicitar production access cedo ou mockar C7 |

## Auditoria de custo (~R$15/mês)

- Lambda/API Gateway ~R$0 ✅ (free tier cobre com folga)
- DynamoDB <R$5 ✅ → **R$0** (free tier permanente, on-demand, TTL)
- EventBridge <R$1 ✅ (Scheduler gratuito na prática)
- CloudWatch <R$5 ✅ → **R$0** com retenção curta (1–7 dias; setar no IaC)
- S3 <R$1 ✅ (5GB free)
- SQS+SES <R$1 ✅ (SQS always-free)
- Cognito R$0 ✅ (5–10 usuários)
- **Secrets Manager — AUSENTE na tabela** ⚠ → **trocar por SSM Parameter Store (SecureString, Standard) = R$0** (Secrets Manager ~R$7–9/mês)
- Step Functions: não listado; ⚠ duplicidade com EventBridge Scheduler — **escolher um** (recomendo Scheduler)

**Total realista**: R$2–15/mês com correções + volume demo moderado; R$60–100/mês se Claude Haiku pago a 5k msgs. Criar **AWS Budgets com alarme (grátis) no dia 1**.

## Viabilidade de compliance (LGPD)

- **LGPD aplica-se à POC** (art. 3º, I e II): tratamento de dados pessoais via Telegram de titulares no Brasil. Sem exceção "POC acadêmica". Resposta Q2 = A+C confirma.
- **Base de imóveis sintética**: sem PII, fora do alcance LGPD. **Mas** conversa no Telegram gera PII real (nome/e-mail/telefone) a partir do primeiro teste com humano.
- **Recomendação principal**: fechar POC 100% em dado sintético (imóveis + leads + CRM) — elimina transferência internacional de PII e reduz LGPD a controles de segurança by design. Maior redução de risco por menor custo.
- **Papéis**: operador da POC = controlador de fato; W Levitt formal só na produção (assumption).
- **Não alegar** "conformidade LGPD" nem "trilha de auditoria LGPD completa" — alegar "segurança e privacidade por design (LGPD)".

## Riscos consolidados

1. **Custo/dependência OpenRouter — ALTO**: SPOF externo; sem crédito = demo morta. Mitigação: crédito pré-carregado, cap de tokens, fallback Bedrock documentado (não budgetado), modelo de reserva barato.
2. **Cold start vs NF-03 <4s — MÉDIO**: warm ping; imports lazy; medir p90 cedo.
3. **Limite de pacote Lambda 250MB — MÉDIO**: RAG (torch), voice (whisper+ffmpeg), anomalias (torch AE). Mitigação transversal: pré-computar/treinar offline, inferência leve, artifacts no S3, layers separadas.
4. **SES sandbox — BAIXO**: production access cedo ou mockar C7.
5. **Streamlit Community Cloud — BAIXO**: sleep/instabilidade; fallback local.
6. **Calibração de anomalias — BAIXO**: dados sintéticos; posicionar como metodologia, não produto.
7. **Região us-east-1**: OK (mais barato, Bedrock disponível; LGPD/residência-BR não exigido na POC); latência ~120–150ms irrelevante contra NF-03. Roadmap reavalia sa-east-1.
8. **Deadline 12/10**: viável; risco é volume de módulos. **MoSCoW sequencial**: núcleo (canal→router→RAG→qualificação→handoff→dashboard) primeiro; follow-up/CRM/voice/anomalias/SES depois; congelar infra ≥1 semana antes da gravação.

## Recomendações infra

- IaC: Terraform (PRD §7.4) mantém-se; teardown `terraform destroy` cobre recursos órfãos.
- Tags de custo (`Project`, `Env=poc`) + AWS Budgets desde o dia 1 — evidência FinOps.
- Log retention explícita; alarme de erro DLQ (1 básico free).
- Secrets: SSM Parameter Store; KMS default (chave gerenciada AWS, R$0); CMK só se requisito de auditoria real.

## Suposições flagadas (conservador)

- Custo por mensagem do OpenRouter; latência de cold start; disponibilidade de produção do SES; comportamento do Streamlit Community Cloud — todas com mitigação barata indicada.
- Custo Claude Haiku pago a 5k msgs ≈ R$85–90/mês (não R$8–15); faixa R$15 só com modelo econômico + volume demo.