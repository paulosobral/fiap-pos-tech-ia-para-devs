## Context

A change `monitoramento-multimodal-pacientes` (task 7, "Finding 2" do review final whole-branch) introduziu `_extract_pose_frame_series` decorada com `@st.cache_data`, cacheando a extração de pose (YOLOv8-pose sobre todos os frames) por conteúdo do vídeo, para que mover os sliders de sensibilidade/zona não reprocessasse o vídeo inteiro. A implementação original criava a `st.progress` bar no chamador (dentro do bloco `if st.button("Processar vídeo"):`) e passava um callback `_on_frame_processed` para dentro da função cacheada, que o invocava para atualizar a barra a cada frame.

Isso funciona na primeira chamada (cache miss, execução real) mas quebra em qualquer chamada subsequente com os mesmos bytes de vídeo (cache hit): o Streamlit registra ("grava") toda chamada de elemento de UI feita durante a execução real de uma função `@st.cache_data`, para reproduzir essas chamadas em hits futuros sem reexecutar a função. A gravação inclui as chamadas de `progress_bar.progress(...)` feitas pelo callback — mas o objeto `progress_bar` referenciado pertence a um bloco de layout criado em uma execução de script *anterior* (a run em que ocorreu o cache miss), que não existe mais na run atual. O replay falha com `CacheReplayClosureError`.

## Goals / Non-Goals

**Goals:**
- Eliminar o `CacheReplayClosureError` ao processar um segundo vídeo ou reprocessar o mesmo vídeo na mesma sessão.
- Preservar o benefício de cache introduzido pelo Finding 2 original (mover sliders não deve reprocessar o vídeo inteiro).
- Preservar a barra de progresso visível durante o processamento real (cache miss).

**Non-Goals:**
- Não estende cache/progress pattern às outras abas (Sinais Vitais, Áudio, Prescrições não usam `st.cache_data` sobre uma função que também emite UI).
- Não altera a lógica de extração de pose em si (`video/pose.py`).

## Decisions

### D1: Mover a `st.progress` bar inteira para dentro da função cacheada
A opção recomendada pela própria mensagem de erro do Streamlit ("Move the creation of $THING inside the function") é a mais simples e correta aqui: como a barra de progresso só existe para dar feedback durante o cache miss (execução real), criá-la e finalizá-la (`.empty()`) inteiramente dentro de `_extract_pose_frame_series` garante que a gravação de replay seja autoconsistente — em um cache hit futuro, o replay recria e imediatamente esvazia a mesma barra, o que é inofensivo (um flash rápido de UI, se perceptível).
**Alternativas consideradas**:
- Remover `@st.cache_data` da função — rejeitada, pois reintroduz o Finding 2 original (reprocessamento completo do vídeo a cada movimento de slider).
- Mover a chamada do elemento Streamlit (a barra) para fora da função cacheada e não repassar callback nenhum, perdendo o feedback de progresso durante o cache miss — rejeitada por regressão de UX (vídeos maiores ficam sem indicação de progresso).

## Risks / Trade-offs

- **[Trade-off] Em um cache hit, a barra de progresso ainda é criada e imediatamente removida** → aceito; é um efeito colateral inofensivo do mecanismo de replay do Streamlit, não uma regressão funcional. Não há chamada de callback por frame nesse caso (a função não executa de fato), então não há impacto de performance.
- **[Risco] Qualquer futura chamada de elemento Streamlit dentro de uma função `@st.cache_data` deve seguir a mesma regra** (criar e mutar inteiramente dentro da função, nunca receber um handle de fora) → mitigação: documentado no docstring da função para evitar recorrência.

## Migration Plan

Não aplicável — correção de bug em código já implementado, sem mudança de schema/dados. Nenhuma ação de usuário necessária.

## Open Questions

Nenhuma pendente.
