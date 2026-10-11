# Resultados de Desempenho — Agente SDR Imobiliário (POC)

**Nenhum teste de carga foi executado.** Este documento registra só o que foi observado.

## Observações (não são medição formal)
- Os logs do router, em conversas reais de teste com LLM real, mostram **8 a 15 s por turno**; o tempo se divide entre a classificação do roteador e a humanização da resposta (duas chamadas de LLM por turno, com fallback entre modelos quando o DeepSeek devolve 429).
- O gate de qualidade com LLM real (62 cenários) leva de 6 a 7 minutos, com oscilações de uma execução para outra.
- Não há p50/p90 calculado, nem decomposição por subsegmento confirmada: o X-Ray ainda não foi visto numa subida real depois da última mudança.

## Conclusão
NFR1.1 **não atingido** nas observações (turnos acima de 10 s ocorrem), e **não verificado** formalmente. NFR1.2 e NFR8.1 sem teste.
