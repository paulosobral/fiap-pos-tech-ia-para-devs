# Detecção de Anomalias — Agente SDR Imobiliário (POC)

## Como roda
- **Lambda `anomaly-detector`**, disparada pelo EventBridge a cada minuto (`var.anomaly_schedule = rate(1 minute)`).
- Extrai características das conversas (`feature_extractor.py`), pontua com o `scorer` (modelo de isolamento, com fallback heurístico quando o lote tem menos de 5 linhas) e declara anomalia quando o score agregado chega a **0,7** (`ANOMALY_THRESHOLD`).
- Os alertas são gravados em `sdr-alerts` (índice `lead-index` por lead) e aparecem no dashboard de negócio. Há um gate de agendamento (`scheduler_gate.py`) que usa esse índice.

## Calibragem
Limiar conservador, para reduzir falso positivo. A revisão do limiar deve olhar os alertas reais acumulados no dashboard. Um teste de qualidade com LLM real (`test_anomaly_e2e_quality.py`) cobre o caso "conversa longa de madrugada gera alerta que chega ao dashboard".

## Lacunas
- Sem notificação ativa (nenhum alarme/SNS): o gestor precisa abrir o dashboard.
- O índice `lead-index` de alertas foi corrigido no Terraform e ainda não foi visto funcionando numa subida real.
