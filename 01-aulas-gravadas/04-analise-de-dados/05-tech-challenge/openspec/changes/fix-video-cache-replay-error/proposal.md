## Why

Ao processar um segundo vídeo (ou reprocessar o mesmo vídeo, incluindo o novo formato mkv adicionado pela change `suporte-video-mkv`), a aba Vídeo lança `streamlit.runtime.caching.cache_errors.CacheReplayClosureError` e a UI quebra. O bug só se manifesta na segunda chamada em diante (cache hit de `st.cache_data`), por isso passou pelo review da change `monitoramento-multimodal-pacientes` sem ser detectado — nenhuma verificação anterior clicou "Processar vídeo" duas vezes.

## What Changes

- Move a criação e atualização da `st.progress` bar para **dentro** de `_extract_pose_frame_series` (a função decorada com `@st.cache_data`), em vez de criá-la no chamador e mutá-la via callback passado para dentro da função cacheada.
- Remove o parâmetro `_on_frame_processed` de `_extract_pose_frame_series` — a barra de progresso agora é totalmente autocontida na função cacheada.
- **BREAKING** (interno, não afeta usuário final): assinatura de `_extract_pose_frame_series` muda (remove `_on_frame_processed`); é uma função privada de `app.py`, sem consumidores externos.

## Capabilities

### New Capabilities
(nenhuma)

### Modified Capabilities
- `video-motion-analysis`: o requirement "Upload e processamento de vídeo clínico" ganha um requisito explícito de que o processamento deve funcionar de forma repetível (múltiplos vídeos na mesma sessão, ou reprocessamento do mesmo vídeo) sem erro — comportamento que já era esperado implicitamente, mas que estava quebrado e não coberto por nenhum scenario.

## Impact

- **Código alterado**: `app.py` (`_extract_pose_frame_series` e o call site na aba Vídeo).
- **Causa raiz**: `st.cache_data` grava (replay) todas as chamadas de elemento Streamlit feitas dentro da função cacheada, para reexibi-las em cache hits futuros. Um elemento (`st.progress`) criado no chamador mas mutado a partir de dentro da função cacheada gera uma gravação inválida — no replay, referencia um bloco de UI de uma execução de script anterior que não existe mais.
- **Testes**: reprodução isolada do bug via `streamlit.testing.v1.AppTest` (2 execuções simulando cache-miss + cache-hit) confirmando a exceção antes da correção e sua ausência depois.
- **Sem impacto** nas demais 3 abas nem no pipeline de extração de pose/análise em si (`video/pose.py`, `video/analysis.py` não mudam).
