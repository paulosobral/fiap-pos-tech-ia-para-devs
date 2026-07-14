## 1. Fixar versão do pyarrow

- [x] 1.1 Fixar `pyarrow<25` em `requirements.txt` (comentário explicando a causa raiz: pyarrow 25.0.0 + scikit-learn importado + rerun do Streamlit)
- [x] 1.2 Reinstalar dependências no `venv/` do projeto (`pip install -r requirements.txt`)
- [x] 1.3 Confirmar que `pyarrow` instalado é da série 24.x — `pyarrow 24.0.0` confirmado; `pandas` permanece `2.2.3` (já instalado, não é afetado por este fix — testado que a versão do pandas é irrelevante para o bug)

Nota: a investigação inicial desta change havia identificado incorretamente `pandas==3.0.3` como causa raiz e fixado `pandas==2.2.3` em vez de `pyarrow`. Reproduzindo o cenário completo do app real (upload de vídeo → YOLOv8-pose → "Processar vídeo", que também importa `scikit-learn` para a aba Sinais Vitais), o crash persistiu mesmo com esse pin. Isolado corretamente: a causa raiz é `pyarrow==25.0.0` em combinação com `scikit-learn` importado + renderização num rerun posterior ao primeiro. `proposal.md`/`design.md` foram reescritos para refletir a investigação completa; o pin final em `requirements.txt` é `pyarrow<25` (pandas não fixado).

## 2. Verificação

- [x] 2.1 Rodar suíte completa de testes (`venv/bin/python -m pytest tests/ -q`) — 107/107 passando, nada quebrou
- [x] 2.2 Reproduzir a renderização de `st.dataframe` via `AppTest` no cenário mínimo (import de `sklearn.ensemble.IsolationForest` + `st.dataframe` dentro de `if st.button`) e confirmar ausência de crash com `pyarrow<25` — confirmado 3/3 execuções sem exceção (antes: 3/3 com `SIGSEGV`/exit 139)
- [x] 2.3 Confirmar boot real do app (`streamlit run app.py`, HTTP 200, sem traceback no log) e exercitar via `AppTest` o fluxo real completo que o usuário reportou como travando: upload de `data/demo_pose_walk.mp4` → clicar "Processar vídeo" → clicar novamente (segundo rerun) — confirmado 2/2 execuções sem exceção, incluindo a renderização real de `st.dataframe(report_df)` com o relatório de desvios do vídeo

## 3. Documentação

- [x] 3.1 Remover a menção à limitação conhecida de segfault do `AppTest`/pyarrow em `README.md` e `RELATORIO_TECNICO.md`
- [x] 3.2 Documentar em `README.md` por que `pyarrow` tem um limite superior fixado neste projeto (causa raiz resumida)
