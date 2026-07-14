## Context

Hoje a aba Vídeo (`app.py`) mostra o slider de sensibilidade imediatamente após o upload, com valor fixo `2.0`, e só extrai a pose (YOLOv8-pose sobre todos os frames, via `_extract_pose_frame_series`, cacheada por conteúdo do vídeo desde a change `fix-video-cache-replay-error`) quando o usuário clica em "Processar vídeo". A extração de pose produz as séries de ângulo/velocidade por frame que alimentam `anomaly.zscore.detect_anomalies` — a mesma informação necessária para calcular quão "ruidoso" é o movimento naquele vídeo específico e, a partir disso, sugerir um threshold de partida mais adequado do que um valor fixo genérico.

Confirmado com o usuário: a extração deve rodar automaticamente ao fazer upload (não atrás de um clique explícito), para que o slider já apareça pré-populado com a sugestão. Isso já é function é cacheada por conteúdo do vídeo, então clicar "Processar vídeo" depois não repete a extração.

## Goals / Non-Goals

**Goals:**
- Ao carregar um vídeo, calcular automaticamente um valor de sensibilidade sugerido, derivado da variação real de ângulo/velocidade daquele vídeo.
- Pré-popular o slider com esse valor sugerido, deixando claro na UI que é uma sugestão para aquele vídeo (não um padrão fixo do sistema).
- Preservar a possibilidade de o usuário ajustar livremente o slider para qualquer valor no intervalo já existente.
- Não duplicar a extração de pose: a mesma chamada cacheada usada para calcular a sugestão deve ser reaproveitada quando o usuário clicar "Processar vídeo".

**Non-Goals:**
- Não estende sugestão automática à zona crítica (área de risco) — depende de conhecimento humano sobre o cenário filmado, não é um cálculo estatístico derivável do vídeo (ver proposal desta change).
- Não altera o algoritmo de detecção de anomalia (`anomaly.zscore.detect_anomalies`) em si — só o valor de partida do parâmetro `threshold` exposto na UI.

## Decisions

### D1: Extrair pose no upload (fora do botão), reaproveitando o cache existente
A extração de pose passa a ser chamada assim que `video_file` está disponível (upload feito e formato validado), antes de renderizar o slider — não mais só dentro do bloco `if st.button("Processar vídeo"):`. Como `_extract_pose_frame_series` já é `@st.cache_data` por bytes do vídeo (change `fix-video-cache-replay-error`), essa chamada "adiantada" no upload e a chamada de dentro do botão "Processar vídeo" (mantida, para o fluxo de análise/alertas em si) resolvem para o mesmo resultado cacheado — só a primeira delas (a do upload) de fato roda o YOLO; a segunda é hit de cache.
**Alternativa considerada**: manter a extração só dentro do botão e adicionar um botão secundário "Sugerir sensibilidade" — rejeitada pelo usuário, que preferiu o cálculo automático sem clique extra.

### D2: Fórmula da sugestão — threshold baseado no desvio padrão combinado de ângulo e velocidade
Sugestão simples e explicável: usar um múltiplo fixo (ex.: um valor calibrado empiricamente, a determinar durante a implementação) como referência de "quantos desvios-padrão de folga" o threshold deveria ter acima do ruído típico já embutido no próprio cálculo de z-score — na prática, um valor de sensibilidade sugerido que produza um número razoável de anomalias (nem zero, nem a maioria dos frames) para o vídeo carregado, calculado por uma heurística simples sobre a distribuição das séries de ângulo/velocidade (ex.: um percentil ou múltiplo do desvio padrão), não por uma otimização custosa.
**Alternativa considerada**: buscar automaticamente o threshold que produz uma taxa-alvo de anomalias (ex.: 5% dos frames) via busca numérica — rejeitada por complexidade desproporcional ao benefício nesta fase; fica como possível evolução futura, documentada como tal.

### D3: UI comunica que o valor é uma sugestão, não uma imposição
O slider mantém total liberdade de ajuste manual (mesmo intervalo/step de hoje); a UI exibe uma mensagem próxima ao slider (ex.: `st.caption`) indicando o valor sugerido calculado para aquele vídeo e que o usuário pode ajustar livremente.

## Risks / Trade-offs

- **[Trade-off] Processamento pesado (YOLO) passa a rodar no upload, sem clique explícito** → aceito pelo usuário; mitigado pelo cache (não duplica processamento ao clicar "Processar vídeo" depois) e por um spinner/feedback visual durante essa extração inicial.
- **[Risco] Vídeo sem nenhuma pessoa detectada em nenhum frame (`has_pose=False` em todos)** → não há série válida para calcular desvio padrão; a sugestão deve degradar graciosamente para o valor fixo atual (`2.0`) nesse caso, sem erro.
- **[Risco] Fórmula de sugestão é uma heurística simples, não uma otimização — pode não ser ideal para todo tipo de vídeo** → aceito conscientemente (D2); documentado como heurística de partida, não como valor "correto" garantido.

## Migration Plan

Não aplicável — mudança de UX em código já implementado, sem dados persistidos a migrar.

## Open Questions

Nenhuma pendente — disparo automático no upload confirmado com o usuário; fórmula exata do cálculo (D2) fica definida durante a implementação com base em uma heurística simples e testável.
