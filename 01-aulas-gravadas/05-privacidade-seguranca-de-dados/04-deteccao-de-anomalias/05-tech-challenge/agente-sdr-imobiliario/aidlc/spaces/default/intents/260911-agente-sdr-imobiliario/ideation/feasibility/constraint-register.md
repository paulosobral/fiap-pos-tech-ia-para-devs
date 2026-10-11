# Constraint Register — Agente SDR Imobiliário (W Levitt)

> Estágio Feasibility & Constraints (Ideation). Classificação: Técnica / Organizacional / Regulatória. Fonte: feasibility-questions (Q1–Q6) + pareceres aws-platform e compliance.

## Técnicas

| ID | Constraint | Origem | Impacto |
|---|---|---|---|
| CT-01 | Limite de pacote Lambda 250MB (descompactado) — PyTorch (RAG/voice/anomalias) estoura | Parecer AWS | Pré-computar/treinar offline; artifacts no S3; layers separadas por função |
| CT-02 | Cold start 1–3s pode estourar NF-03 (<4s primeira resposta) | Parecer AWS | Warm ping antes/durante demo; imports lazy; medir p90 |
| CT-03 | OpenRouter é SPOF externo; sem crédito = demo morta | Parecer AWS | Crédito pré-carregado; cap de tokens; fallback Bedrock documentado (não budgetado) |
| CT-04 | Custo estimado R$8–15 só fecha com modelo econômico (DeepSeek V3) + volume demo controlado | Parecer AWS | Orçar R$15 com modelo econômico; Haiku = upgrade documentado |
| CT-05 | Secrets Manager custa ~R$7–9/mês (3–4 secrets) | Parecer AWS | Usar SSM Parameter Store (SecureString, Standard) = R$0 |
| CT-06 | Duplicidade EventBridge Scheduler × Step Functions para follow-up | Parecer AWS | Escolher um (recomendo Scheduler one-off) |
| CT-07 | SES em conta nova fica em sandbox (só remetentes verificados) | Parecer AWS | Solicitar production access cedo ou mockar C7 |
| CT-08 | Streamlit Community Cloud dorme por inatividade / infra compartilhada | Parecer AWS | Abrir antes da gravação; fallback local; `st.login` exige Streamlit ≥1.42 |
| CT-09 | OAuth/PKCE do MCP HubSpot não funciona em Lambda assíncrona | Parecer AWS | MCP só na demo (Inspector); adapter usa Private App REST |
| CT-10 | Região us-east-1 (mais barato, Bedrock disponível; residência-BR não exigido na POC) | Parecer AWS | OK na POC; roadmap reavalia sa-east-1 |
| CT-11 | CloudWatch retenção default "never expire" vira custo lento | Parecer AWS | Setar retenção curta (1–7 dias) no IaC |

## Organizacionais

| ID | Constraint | Origem | Impacto |
|---|---|---|---|
| CO-01 | Deadline 12/10 (hackathon FIAP); ideal terminar antes para gravação do vídeo pelo usuário | Q4 | MoSCoW sequencial; congelar infra ≥1 semana antes da gravação |
| CO-02 | Orçamento baixo custo (~R$15/mês POC) | Q4/D | AWS Budgets + alarme no dia 1; SSM no lugar de Secrets Manager |
| CO-03 | Nenhum bloqueador organizacional — projeto novo, sem sistemas legados críticos em produção | Q5-A | Sem freeze; sem dependência de aprovação W Levitt para dados reais (POC usa sintéticos) |
| CO-04 | Time com experiência em Python e AWS (capacidade de build) | Q3-D | Stack Python + AWS serverless viável sem curva de aprendizado |
| CO-05 | Base de imóveis simulada (JSON+S3), sem portal real | Q1-B | Sem integração com ERP/portal |
| CO-06 | Canal Telegram na POC; WhatsApp no roadmap | Q1-C | Escopo POC limitado ao Telegram |
| CO-07 | Conta AWS dedicada/nova para a POC | Q6-A | Provisionar nesta etapa; teardown `terraform destroy` pós-POC |

## Regulatórias (LGPD)

| ID | Constraint | Origem | Impacto |
|---|---|---|---|
| CR-C1 | Masking de PII (nome/e-mail/telefone/CNPJ) antes de qualquer envio a LLM, qualquer provedor | Parecer compliance | Testes unitários com placeholders + scan de saída |
| CR-C2 | Captura e persistência de consentimento (timestamp + texto do aviso + id do canal) | Parecer compliance | Item no DynamoDB |
| CR-C3 | Aviso de privacidade curto na 1ª mensagem e no dashboard | Parecer compliance | Transparência (art. 9º) |
| CR-C4 | Retenção: TTL 90d conversas + TTL definido p/ store PII + exclusão de lead a pedido (manual documentado) | Parecer compliance | Minimização (art. 6º, V) + direitos do titular (art. 18) |
| CR-C5 | Logs CloudWatch com masking de PII | Parecer compliance | Segurança (art. 46) |
| CR-C6 | Demo e testes exclusivamente com dados sintéticos (incl. HubSpot) | Parecer compliance | Elimina transferência internacional de PII |
| CR-C7 | Segredos via SSM/Secrets Manager; zero credenciais hardcoded | Parecer compliance | Segurança (art. 46) |
| CR-C8 | Teardown pós-POC (`terraform destroy`) + descarte de dados; evidência no relatório de verificação | Parecer compliance | LGPD + recursos órfãos |

## Fora do escopo da POC (roadmap produção)

- Bedrock em sa-east-1 (residência Brasil, guardrails nativos)
- DPA formal com provedores; registro de operações de tratamento (art. 37)
- Política formal de retenção; papel de DPO/encarregado
- Trilha de auditoria imutável (POC tem apenas "registro e visibilidade de operações")