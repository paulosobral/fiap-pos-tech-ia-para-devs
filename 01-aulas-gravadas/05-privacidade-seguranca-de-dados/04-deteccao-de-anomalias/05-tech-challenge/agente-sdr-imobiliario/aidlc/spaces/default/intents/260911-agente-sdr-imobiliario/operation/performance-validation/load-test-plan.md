# Plano de Teste de Carga — Agente SDR Imobiliário (POC)

## Alvos (da `build-and-test/performance-test-instructions.md`)
- **NFR1.1**: primeira resposta < 10 s (p50 e p90).
- **NFR1.2**: atendimento simultâneo sem fila.
- **NFR8.1**: escala horizontal automática.

## Perfil de tráfego esperado
Hackathon e demonstração: poucos leads simultâneos (< 10 sessões), conversas curtas, picos só durante a demo. Não há tráfego de produção.

## Plano (não executado neste POC)
1. Medir o turno de ponta a ponta com LLM real: tempo Telegram → router → LLM → resposta, a partir dos subsegmentos `llm:<modelo>` do X-Ray e do log `roteador:`.
2. Carga sintética de 10 sessões concorrentes enviando mensagens ao `POST /webhook/telegram` (com o segredo do webhook), medindo p50/p90/p95.
3. Repetir com o tier primário da LLM (DeepSeek) em 429 forçado, para ver o custo do fallback.
4. Gargalos prováveis: latência do provedor de LLM, o router em **1 task Fargate** (256 CPU / 512 MB) e o limite de taxa do OpenRouter.

## Por que não foi executado
O ambiente é efêmero e pago; uma carga com LLM real consome crédito do OpenRouter, e o deploy só ocorre pelo `start.sh`, rodado pelo responsável. Fica como evolução pós-POC.
