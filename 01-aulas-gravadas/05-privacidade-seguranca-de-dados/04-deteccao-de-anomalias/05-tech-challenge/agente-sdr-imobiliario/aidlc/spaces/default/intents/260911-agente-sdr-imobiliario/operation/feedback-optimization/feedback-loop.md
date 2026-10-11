# Ciclo de Feedback — Agente SDR Imobiliário (POC)

## Fontes de feedback
1. **Conversas reais de teste** com o bot: as falhas encontradas viraram correções e cenários no gate de qualidade (`tests/quality`, hoje 62 cenários com LLM real).
2. **Dashboard de negócio**: funil, leads, imóvel escolhido, valor e alertas de anomalia.
3. **X-Ray e logs `roteador:`**: decisões do roteador e latência por etapa.
4. **Ritual de aprendizados do AI-DLC**: decisões do responsável viram regras de memória do projeto.
5. **Retorno do corretor sobre os leads** no HubSpot: ainda sem canal formal.

## Como o feedback vira melhoria
Problema observado → causa investigada → correção no código ou no prompt → teste (unidade ou cenário com LLM real) → ADR em `decisions.md` quando a decisão é arquitetural → documentação no PRD e no documento de entrega.

## Próximas melhorias priorizadas (do que ficou em aberto)
1. Medir o p90 e reduzir o tempo do turno (NFR1.1).
2. Emitir métricas e alarmes (NFR5.3).
3. Filtrar o follow-up para não reativar leads já encaminhados.
4. Garantir que a lista que o lead vê seja a mesma que o bot guardou.
5. Pergunta natural de orçamento quando o lead não informa.
