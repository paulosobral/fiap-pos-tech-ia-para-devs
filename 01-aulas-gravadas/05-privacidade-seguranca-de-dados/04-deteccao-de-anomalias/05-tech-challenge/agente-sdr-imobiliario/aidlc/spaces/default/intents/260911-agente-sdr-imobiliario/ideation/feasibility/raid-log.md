# RAID Log — Agente SDR Imobiliário (W Levitt)

> Estágio Feasibility & Constraints (Ideation). Risks, Assumptions, Issues, Dependencies. Fonte: pareceres aws-platform e compliance + feasibility-questions.

## Risks

| ID | Risco | Prob/Impacto | Mitigação |
|---|---|---|---|
| R-01 | Dependência/custo OpenRouter (SPOF externo; sem crédito = demo morta) | Alta / Alta | Crédito pré-carregado; cap de tokens; fallback Bedrock documentado (não budgetado); modelo de reserva barato |
| R-02 | Cold start vs NF-03 (<4s primeira resposta) | Média / Média | Warm ping antes/durante demo; imports lazy; medir p90 real cedo |
| R-03 | Limite de pacote Lambda 250MB (torch: RAG/voice/anomalias) | Média / Média | Pré-computar/treinar offline; inferência leve; artifacts no S3; layers separadas |
| R-04 | PII real trafegando para OpenRouter sem salvaguarda contratual (sem DPA alinhado LGPD) | Média / Média | CR-C6 (dado sintético end-to-end); residual aceitável p/ contexto acadêmico |
| R-05 | Alegação de "conformidade LGPD" além do implementado | Baixa / Média | Wording "segurança e privacidade por design (LGPD)"; não alegar compliance formal |
| R-06 | SES sandbox (conta nova) bloqueia ingestão C7 | Baixa / Média | Solicitar production access cedo ou mockar C7 |
| R-07 | Streamlit Community Cloud dorme/instável na gravação | Baixa / Baixa | Abrir antes; fallback local (mesma imagem) |
| R-08 | Calibração de anomalias com dados sintéticos ("anomalia" = o que foi plantado) | Baixa / Baixa | Posicionar como metodologia/diferencial, não detector de produção |
| R-09 | Escopo (16 módulos) contra deadline 12/10 | Média / Alta | MoSCoW sequencial; congelar infra ≥1 semana antes da gravação |

## Assumptions

| ID | Assumption | Status |
|---|---|---|
| A-01 | Operador da POC = controlador de fato (define finalidade/meios); W Levitt formal só na produção | Aberta |
| A-02 | LGPD incide mesmo em POC acadêmica (leitura conservadora do art. 3º) | Aberta |
| A-03 | Custo por mensagem do OpenRouter e latência de cold start comportam a faixa R$15 | Aberta |
| A-04 | Base de imóveis sintética não reproduz/deriva PII real (origem verificada) | Aberta |
| A-05 | Bedrock fallback é contingência documentada, não item de budget POC | Aberta |
| A-06 | Nenhum bloqueador organizacional (projeto novo, sem legados críticos) | Confirmada (Q5-A) |
| A-07 | Time com capacidade de build Python + AWS | Confirmada (Q3-D) |

## Issues

| ID | Issue | Status |
|---|---|---|
| I-01 | Falta caminho de exercício de direitos do titular na POC (deleção manual documentada fecha o mínimo) | Aberta |
| I-02 | Evidência de consentimento precisa de forma demonstrável (timestamp, texto, canal/id, versão do aviso) para KPI "testes LGPD" | Aberta |
| I-03 | Configurações de privacidade do provedor são manuais (logging off OpenRouter, ZDR Anthropic) — passo esquecível | Aberta |
| I-04 | Trilha de auditoria em nível "consulta simples" (timeline DynamoDB) não caracteriza auditoria LGPD (art. 37) | Aberta — não alegar no pitch |

## Dependencies

| ID | Dependência | Dono |
|---|---|---|
| D-01 | Configuração OpenRouter (logging off) + ZDR Anthropic — checklist de setup | Setup POC |
| D-02 | Crédito OpenRouter pré-carregado antes da gravação | Usuário |
| D-03 | AWS Budgets + alarme criados no dia 1 | Setup POC |
| D-04 | SES production access (ou decisão de mockar C7) | Setup POC |
| D-05 | Verificação da origem dos dados sintéticos (sem PII real) | Setup POC |