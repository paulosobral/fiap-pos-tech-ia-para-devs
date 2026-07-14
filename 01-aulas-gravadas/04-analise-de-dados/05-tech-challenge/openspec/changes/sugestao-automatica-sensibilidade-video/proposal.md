## Why

O slider de sensibilidade da aba Vídeo abre com um valor fixo (`2.0`) igual para qualquer vídeo, mas o quão "normal" é a variação de ângulo/velocidade de um paciente depende do próprio vídeo (tipo de movimento, ruído do rastreamento de pose, framerate). Um valor fixo pode ser inadequado tanto por ser sensível demais (gerando ruído de alertas em vídeos com movimento naturalmente mais variável) quanto por ser pouco sensível (deixando passar desvios reais em vídeos mais estáveis). Como a extração de pose já é feita e cacheada por conteúdo do vídeo (ver change `fix-video-cache-replay-error`), é possível usar a variação estatística real das séries de ângulo/velocidade daquele vídeo específico para sugerir um valor de partida mais adequado, sem exigir nenhuma configuração manual do usuário.

## What Changes

- Adiciona uma sugestão automática de sensibilidade calculada a partir da extração de pose já realizada: o slider passa a abrir pré-populado com um valor sugerido para o vídeo carregado, em vez do valor fixo `2.0`.
- O usuário continua podendo ajustar o slider manualmente para qualquer valor no intervalo existente — a sugestão é só o ponto de partida, não um valor travado.
- A UI indica de forma visível que aquele é um valor sugerido para o vídeo carregado (não um padrão genérico), para não confundir com uma imposição do sistema.
- Isso muda a ordem de operações da aba: hoje o slider aparece antes de qualquer processamento; a sugestão só pode ser calculada depois que o vídeo é decodificado e a pose extraída (mesma extração já cacheada), então o fluxo passa a: usuário faz upload → sistema extrai pose e calcula a sugestão automaticamente → slider aparece já pré-populado com a sugestão → usuário clica "Processar vídeo" (ou ajusta o slider antes de clicar).

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `video-motion-analysis`: o requirement "Detecção de anomalia postural com sensibilidade ajustável" ganha um novo scenario cobrindo a sugestão automática de sensibilidade calculada a partir do vídeo carregado.

## Impact

- **Código alterado**: `app.py` (aba Vídeo — reordena a extração de pose para antes da exibição do slider, calcula a sugestão); possivelmente um novo helper em `video/analysis.py` para o cálculo estatístico da sugestão (ex.: baseado no desvio padrão das séries de ângulo/velocidade).
- **UX**: extração de pose (chamada de YOLOv8 sobre todos os frames) passa a ocorrer no upload do vídeo, não só ao clicar "Processar vídeo" — para vídeos grandes, isso adianta um processamento que hoje só acontecia depois do clique. Mitigado pelo cache já existente (mesma extração não roda de novo ao clicar "Processar vídeo" depois).
- **Sem impacto** nas demais 3 abas.
