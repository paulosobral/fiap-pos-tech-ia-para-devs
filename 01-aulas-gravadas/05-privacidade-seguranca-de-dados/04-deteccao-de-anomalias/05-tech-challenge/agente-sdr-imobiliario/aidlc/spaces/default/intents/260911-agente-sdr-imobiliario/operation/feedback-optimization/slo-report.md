# Relatório de SLO — Agente SDR Imobiliário (POC)

Não há dados de produção: o ambiente é efêmero e de demonstração. O relatório compara as metas de `observability-setup/slo-config.md` com o que foi observado nos testes.

| SLO | Meta | Observado | Situação |
|---|---|---|---|
| Resposta rápida (p90) | < 10 s | 8–15 s por turno nos testes com LLM real; sem medição formal | Não atingido em parte dos turnos |
| Disponibilidade na janela | 99% | Não medida (ambiente destruído a cada sessão) | Sem dado |
| Mensagens sem perda | 0 em DLQ | Sem registro de perda nos testes manuais | Sem dado formal |
| Lead chega ao CRM | 95% | HubSpot validado de ponta a ponta nos testes; sem taxa medida | Sem dado formal |

## Ações
1. Medir o p90 pelo X-Ray numa subida real (ver `performance-validation/nfr-validation-matrix.md`).
2. Reduzir o tempo do turno: uma chamada de LLM a menos ou modelo mais rápido na humanização.
