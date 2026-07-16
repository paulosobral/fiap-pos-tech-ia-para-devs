## Context

`anomaly/zscore.py::detect_anomalies(series, window, threshold)` computa, para cada ponto, o z-score contra a média e o desvio-padrão populacional (`std(ddof=0)`) de uma janela móvel de tamanho `window` que **inclui o próprio ponto**. Propriedade matemática: numa janela de `n` pontos onde um é um outlier extremo e os outros iguais, o z-score do outlier tende a `sqrt(n-1)` (limite quando o outlier → ∞). Logo `|z|` NUNCA excede `sqrt(window-1)`, independentemente de quão grande seja a anomalia.

A aba Sinais Vitais usa `DEFAULT_WINDOW=6`, `DEFAULT_THRESHOLD=3.0` (`vital_signs/analysis.py`). Como `sqrt(6-1)=2.236 < 3.0`, a camada z-score marca ZERO em qualquer entrada (verificado: um pico de FC=100000 dá 0 flags). Só o Isolation Forest detecta algo; `alta_confianca` (concordância das duas camadas) é inatingível. Os controles manuais da aba (`st.number_input` de janela e threshold) permitem ao usuário recriar o mesmo problema sem aviso.

Nota: as abas Vídeo e Áudio têm seus próprios `DEFAULT_WINDOW` (10 e 5) e não estão em escopo — o fix é focado em Sinais Vitais, que é onde o default quebra e onde o usuário topou com o problema.

## Goals / Non-Goals

**Goals:**
- Fazer a camada z-score de Sinais Vitais efetivamente detectar anomalias nos valores padrão.
- Impedir que combinações janela/threshold inefetivas (threshold ≥ `sqrt(window-1)`) passem despercebidas nos controles manuais — avisar o usuário.

**Non-Goals:**
- Não alterar a matemática de `anomaly/zscore.py` (ddof=0, janela inclusiva) — é compartilhada por Vídeo/Áudio; mudá-la exigiria re-validar as três abas e está fora do escopo deste bugfix pontual.
- Não mexer nos defaults de Vídeo/Áudio.
- Não impedir o usuário de escolher manualmente uma combinação inefetiva — só avisá-lo (liberdade preservada, com transparência).

## Decisions

### D1: `DEFAULT_WINDOW` de Sinais Vitais 6 → 13
Com `threshold=3.0` fixo, o menor `window` cujo teto `sqrt(window-1)` supera 3.0 é 11 (`sqrt(10)=3.162`), mas a folga é mínima — um pico precisa ser quase infinito pra chegar perto do teto, e picos clínicos reais ficariam raspando. `window=13` dá teto `sqrt(12)=3.464`, folga confortável: verificado na amostra MIMIC real, `window=13, threshold=3.0` marca 20 leituras (de 2400) — anomalias reais, não ruído. Escolhido 13 por robustez. `DEFAULT_THRESHOLD` permanece 3.0 (valor estatisticamente convencional para "outlier").
**Alternativa considerada**: baixar `DEFAULT_THRESHOLD` para ~2.0 mantendo window=6 — rejeitada; 2.0 é um outlier fraco (marca muito ruído), e a raiz do problema é a janela curta demais para o threshold, não o threshold.

### D2: Aviso de combinação inefetiva na UI
Helper puro em `vital_signs/analysis.py`, ex.: `zscore_threshold_is_reachable(window, threshold) -> bool` (retorna `threshold < sqrt(window-1)`). A aba, ao ler os `number_input` de janela e threshold, se a combinação for inefetiva, exibe um `st.warning` explicando que nenhuma anomalia z-score é detectável nesses valores (o `|z|` máximo possível é `sqrt(window-1)`) e sugerindo aumentar a janela ou baixar o threshold. Assim o default fica correto E o usuário não recria o bug silenciosamente pelos controles.
**Alternativa considerada**: clampar/forçar valores válidos nos inputs — rejeitada; tira liberdade do usuário e é menos transparente que um aviso explicativo.

## Risks / Trade-offs

- **[Trade-off] Janela maior (13) suaviza mais** → aceito; 13 leituras ainda é uma janela local (13h no dataset horário), e o ganho (camada z-score finalmente funcional) supera. A alternativa (janela curta) é literalmente não-funcional.
- **[Risco] O teto `sqrt(window-1)` é uma propriedade de `anomaly/zscore.py` não corrigida aqui** → aceito e documentado; o aviso da UI (D2) cobre o caso geral (qualquer window/threshold que o usuário escolher), então o problema não volta silenciosamente mesmo mantendo a matemática atual. Uma correção de fundo do z-score fica como possível trabalho futuro, fora deste escopo.

## Migration Plan

Não aplicável — mudança de parâmetro padrão + aviso de UI, sem dados/persistência.

## Open Questions

Nenhuma — `window=13` e a abordagem de aviso (não clamp) fechadas.
