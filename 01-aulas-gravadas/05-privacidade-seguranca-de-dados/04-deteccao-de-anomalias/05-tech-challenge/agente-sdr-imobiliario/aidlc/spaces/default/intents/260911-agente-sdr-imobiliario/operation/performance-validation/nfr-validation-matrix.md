# Matriz de Validação de NFR — Desempenho (POC)

| NFR | Meta | Evidência | Status | Próximo passo |
|---|---|---|---|---|
| NFR1.1 | Primeira resposta < 10 s (p50/p90) | Observação de logs: 8–15 s por turno com LLM real; sem p50/p90 | **Não atingido / não verificado** | Medir pelo X-Ray; reduzir chamadas de LLM por turno ou usar modelo mais rápido na humanização |
| NFR1.2 | Atendimento simultâneo sem fila | Sem teste de concorrência; router em 1 task Fargate | **Não verificado** | Carga sintética de 10 sessões |
| NFR8.1 | Escala horizontal automática | Lambdas e API Gateway escalam por natureza; ECS tem só liga/desliga agendado (`aws_appautoscaling_scheduled_action`), sem política por carga | **Parcial** | Política de escala por CPU/requisições no router, se houver tráfego real |
| NFR5.3 | Métricas de negócio e de desempenho | KPIs de negócio no dashboard; `ResponseTimeP90` e custo nunca emitidos | **Parcial** | Emitir métricas (ver `observability-setup/dashboards.md`) |

## Decisão
Desvios aceitos como escopo do POC acadêmico pelo responsável do projeto, registrados aqui como risco conhecido: a primeira resposta pode passar de 10 s em alguns turnos.
