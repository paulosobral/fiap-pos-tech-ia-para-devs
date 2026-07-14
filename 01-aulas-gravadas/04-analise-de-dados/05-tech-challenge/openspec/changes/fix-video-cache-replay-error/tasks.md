## 1. Corrigir CacheReplayClosureError no reprocessamento de vídeo

- [x] 1.1 Mover criação/atualização/finalização da `st.progress` bar para dentro de `_extract_pose_frame_series` (`app.py`), removendo o parâmetro `_on_frame_processed`
- [x] 1.2 Atualizar o call site na aba Vídeo (`app.py`) para não criar mais a barra de progresso externamente
- [x] 1.3 Reproduzir o bug isoladamente via `streamlit.testing.v1.AppTest` (2 execuções simulando cache-miss + cache-hit) antes da correção, confirmando a exceção
- [x] 1.4 Confirmar ausência da exceção com o mesmo repro após a correção
- [x] 1.5 Rodar suíte completa de testes (`venv/bin/python -m pytest tests/ -q`) e confirmar boot do app (`streamlit run app.py`) sem erros
- [x] 1.6 Persistir o repro como teste de regressão permanente (`tests/test_app_cache_replay_regression.py`), validado nos dois sentidos: reproduz `CacheReplayClosureError` com o padrão antigo (progress bar externa + callback) e passa com o padrão corrigido (tudo dentro da função cacheada)
