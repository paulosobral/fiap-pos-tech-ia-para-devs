# SLOs e SLIs — Agente SDR Imobiliário (POC)

Metas da POC, medidas hoje por inspeção manual; nenhuma é monitorada automaticamente.

| SLO | SLI | Meta | Janela | Estado |
|---|---|---|---|---|
| Resposta rápida (NFR1.1) | tempo entre a mensagem do lead e a resposta do bot, p90 | < 10 s | 24 h de operação | Não atingido hoje: os logs mostram 8–15 s por turno, com LLM real; não medido formalmente |
| Disponibilidade na janela | tasks do router em execução durante 09:00–18:00 BRT | 99% | 7 dias | Não medido; o ambiente é efêmero |
| Mensagens não perdidas (NFR4.1) | mensagens em DLQ sem tratamento | 0 | contínuo | Verificação manual |
| Lead capturado chega ao CRM | leads em handoff com contato que aparecem no HubSpot | 95% | 7 dias | Verificação manual (fila `crm`) |

O orçamento de erro e a medição automática ficam para o estágio `performance-validation` (latência) e para a evolução pós-POC (alarmes).
