# Performance Test Instructions — Agente SDR Imobiliário (POC)

Contexto: NFR1 — **NFR1.1** primeira resposta < 10 s (KPI do PRD); **NFR1.2** atendimento simultâneo sem fila.

## Metas e método
| Target | Meta | Como medir | Ambiente necessário |
|---|---|---|---|
| NFR1.1 | Primeira resposta < 10 s p50/p90 | Instrumentar o caminho webhook → resposta: medir t(telegram receive) → t(stt) → t(llm 1º turno) → t(resposta) | Ambiente integrado com LLM real (OpenRouter) e rede — **performance-validation** |
| NFR1.2 | N sessões simultâneas sem degradação de fila | Carga sintética N≥10 sessões concorrentes; p95 de latência estável | Ambiente integrado + concorrência — **performance-validation** |

## O que é executável localmente (e limites)
- **Benchmark determinístico de pipeline** (sem LLM externo): latência de `pipeline.parse → qualification → flow.step → response` com inputs sintéticos. Serve como **guarda de regressão** de lógica, **não** prova o alvo de < 10 s (o dominante é latência de LLM/rede).
- Não é possível reproduzir p90 e2e localmente sem chamada real ao provedor; o alvo **permanece Unverified neste estágio**, com evidência esperada em `performance-validation` (stage do plano do projeto).

## Comando de guarda local (opcional, informativo)
```bash
.venv/bin/python - <<'PY'
import time, json
from apps.conversation_router.service.flow.sales_flow import *  # pipeline local com stubs
PY
# (a suíte já exercita o pipeline; um harness dedicado pode ser adicionado em performance-validation)
```

## Detecção de regressão
- Em CI (`ci-pipeline`), comparar tempo total das suítes como proxy grosseiro; a medida real p50/p90 fica em `performance-validation`.

## Registro de deferral
- `NFR1.1`, `NFR1.2` → **Owning Stage: performance-validation** (stage agendado no plano; evidência esperada: relatório p50/p90 com LLM real).
