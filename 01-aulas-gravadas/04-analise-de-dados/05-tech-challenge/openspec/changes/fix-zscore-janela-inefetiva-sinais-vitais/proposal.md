## Why

A camada de rolling z-score da aba Sinais Vitais nunca dispara nos valores padrão. `DEFAULT_WINDOW=6` e `DEFAULT_THRESHOLD=3.0`, mas o z-score usado (`anomaly/zscore.py`) computa média/desvio-padrão populacional (ddof=0) sobre uma janela que **inclui o próprio ponto** — o que limita o `|z|` máximo possível a `sqrt(window-1)`. Para `window=6` isso é `sqrt(5) ≈ 2.24`, sempre abaixo de `3.0`. Consequência: mesmo um pico absurdo (ex.: FC = 100000) resulta em ZERO detecções da camada z-score; só o Isolation Forest marca algo. Isso deixa metade da detecção de sinais vitais morta no default e torna o rótulo "alta confiança" (concordância das duas camadas) inatingível — exatamente o contraste que a aba se propõe a demonstrar. Descoberto ao tentar criar amostras de demonstração com anomalias: nenhuma amostra dispara a camada z-score nos defaults, por impossibilidade matemática, não por falta de sinal.

## What Changes

- Ajustar o `DEFAULT_WINDOW` da aba Sinais Vitais para um valor cujo teto de `|z|` (`sqrt(window-1)`) fique com folga acima do `DEFAULT_THRESHOLD` (3.0): **`window=13`** dá teto `sqrt(12) ≈ 3.46`, permitindo que picos reais cruzem o threshold. (`window=11` daria `3.16`, folga apertada; `13` é mais robusto.)
- Adicionar um aviso na UI da aba Sinais Vitais quando o usuário escolher manualmente uma combinação janela/threshold matematicamente inefetiva (threshold ≥ `sqrt(window-1)`), explicando que nenhuma anomalia z-score é detectável com esses valores — para não reintroduzir o problema silenciosamente via os controles manuais.
- **BREAKING** (comportamento da aba, não spec externa): a detecção z-score de sinais vitais passa a de fato marcar anomalias nos valores padrão (antes marcava zero).

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `vital-signs-monitoring`: o requirement de detecção por rolling z-score passa a garantir que os parâmetros padrão da aba são efetivos (capazes de marcar anomalias) e que combinações janela/threshold inefetivas são sinalizadas ao usuário.

## Impact

- **Código alterado**: `vital_signs/analysis.py` (`DEFAULT_WINDOW` 6→13); `app.py` (aviso na aba Sinais Vitais para combinação janela/threshold inefetiva).
- **Sem mudança** em `anomaly/zscore.py` — a matemática do z-score (ddof=0, janela inclusiva) é preservada; só ajustamos o parâmetro padrão que a usa e alertamos sobre combinações ruins. (Não mexer no módulo compartilhado evita re-testar Vídeo e Áudio, que têm seus próprios `DEFAULT_WINDOW` e não estão em escopo aqui.)
- **Testes**: novos cobrindo que o default é efetivo (um pico claro é detectado com `DEFAULT_WINDOW`/`DEFAULT_THRESHOLD`), e o helper/aviso de combinação inefetiva.
- **Habilita** a change `amostras-demo-sinais-vitais` (bloqueada por este bug) a produzir uma amostra com anomalias que dispara as duas camadas nos defaults.
- **Sem impacto** nas outras abas nem em dependências.
