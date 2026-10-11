# Feedback & Optimization Questions

> Respostas derivadas do estado real do projeto; não há dados de produção.

## Q1 — Fonte de dados do relatório de SLO
- A. **Observações dos testes e dos logs**; sem medição de produção (Recommended)
- B. Esperar dados de produção

[Answer]: A

## Q2 — Foco da análise de custo
- A. **Vetores de custo e controles já adotados**, sem faturamento medido (Recommended)
- B. Estimativa em dólares por cenário de tráfego

[Answer]: A

## Q3 — Detecção de drift
- A. **`terraform plan` e `tests/infra` em ambiente efêmero**, mais o registro das divergências de design (Recommended)
- B. AWS Config / drift detection contínuo

[Answer]: A

## Q4 — Ciclo de melhoria
- A. **Teste/conversa real → correção → cenário no gate de qualidade → ADR/documentação**, com lista priorizada do que está em aberto (Recommended)
- B. Painel de feedback do corretor — sem canal para isso ainda

[Answer]: A

---

## Consolidated Summary Confirmation

Resumo do estágio Feedback & Optimization:
1. Relatório de SLO baseado em observações; p90 não atingido em parte dos turnos; sem dados de produção.
2. Custo: vetores e controles (tiering de LLM, tasks pequenas, janela 09–18h, sem NAT/ALB); sem faturamento medido.
3. Drift: ambiente efêmero, sem acúmulo; divergências do design já registradas.
4. Ciclo de feedback e 5 melhorias priorizadas em aberto.

- Looks correct
- Request changes

[Answer]: Looks correct
