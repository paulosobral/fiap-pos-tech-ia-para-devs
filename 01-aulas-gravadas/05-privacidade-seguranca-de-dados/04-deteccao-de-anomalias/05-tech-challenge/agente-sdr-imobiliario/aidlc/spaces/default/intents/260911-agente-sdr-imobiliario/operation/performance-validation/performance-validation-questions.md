# Performance Validation Questions

> Respostas derivadas das observações de logs e da decisão de não gastar crédito de LLM em carga sintética no POC.

## Q1 — Padrão de tráfego
- A. **Demonstração com poucos leads simultâneos** (< 10), sem pico de produção (Recommended)
- B. Tráfego de produção de uma imobiliária

[Answer]: A

## Q2 — Percentis alvo
- A. **p50 e p90 < 10 s** na primeira resposta (NFR1.1) (Recommended)
- B. p95 e p99 também — sem volume para estimar

[Answer]: A

## Q3 — Vazão necessária
- A. **Até 10 sessões simultâneas** (NFR1.2), sem meta de mensagens por segundo (Recommended)
- B. Meta de mensagens por segundo

[Answer]: A

## Q4 — Gargalos prováveis
- A. **Latência da LLM (duas chamadas por turno), limite de taxa do OpenRouter e o router em 1 task Fargate** (Recommended)
- B. DynamoDB/SQS — gerenciados e fora da suspeita

[Answer]: A

## Q5 — Executar a carga agora?
- A. **Não**: documentar o plano e o status real (não atingido / não verificado) e deixar a execução para a evolução pós-POC (Recommended)
- B. Executar a carga com LLM real agora — consome crédito e exige uma subida do ambiente pelo responsável

[Answer]: A

---

## Consolidated Summary Confirmation

Resumo do estágio Performance Validation:
1. Plano de carga documentado, mas **nenhum teste executado**.
2. Observação de logs: 8–15 s por turno; NFR1.1 (< 10 s) não atingido e não verificado formalmente; NFR1.2 sem teste.
3. NFR8.1 parcial (Lambdas escalam; o ECS só liga e desliga por agenda); NFR5.3 parcial.
4. Desvios aceitos como risco conhecido do POC; medição pelo X-Ray fica para depois.

- Looks correct
- Request changes

[Answer]: Looks correct
