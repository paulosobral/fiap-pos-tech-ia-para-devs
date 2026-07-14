## Context

`requirements.txt` não fixava `pyarrow` (dependência transitiva do Streamlit e do pandas), então o `pip install` resolveu para `pyarrow==25.0.0`. Essa versão tem um bug real de segfault (`SIGSEGV`, `libarrow.so.2500`, dentro de `pyarrow/pandas_compat.py::convert_column`, chamado por `streamlit/dataframe_util.py::convert_pandas_df_to_arrow_bytes`, chamado por `st.dataframe`/`st.table`) — reproduzido tanto em uso real (`streamlit run app.py`, relatado pelo usuário) quanto em teste automatizado (`streamlit.testing.v1.AppTest`, já era uma limitação conhecida documentada na change `monitoramento-multimodal-pacientes`).

**Investigação (histórico da causa raiz nesta change):** a hipótese inicial era que `pandas==3.0.3` (que o `pip` também havia resolvido, por `requirements.txt` declarar só `pandas>=2.2.0`) fosse a causa — um teste isolado mínimo (script de duas linhas sem `scikit-learn`/`torch`) parou de crashar ao fixar `pandas==2.2.3`. Essa hipótese estava **incompleta**: ao tentar reproduzir o cenário real do app (upload de vídeo → clicar "Processar vídeo", que usa YOLOv8-pose via `ultralytics`/`torch`), o crash persistiu mesmo com `pandas==2.2.3`. Isolando ainda mais, o fator determinante não era `pandas` nem `torch` — era `scikit-learn` estar importado no processo (só o `import`, `from sklearn.ensemble import IsolationForest`, sem nem instanciar/treinar nada) **combinado com** a renderização de `st.dataframe` ocorrendo num script-rerun do Streamlit posterior ao primeiro (ex.: dentro de `if st.button(...):`, que só executa a partir do segundo rerun em diante). Fixando `pyarrow<25` (resolve para a série 24.x), o mesmo cenário mínimo — e o cenário real completo do app, incluindo upload de vídeo, YOLOv8-pose e clique duplo em "Processar vídeo" — para de crashar, independente da versão do pandas (testado com `pandas==2.2.3` e `pandas==3.0.3`, ambas funcionam com `pyarrow<25`).

Isso é consistente com o app usar `scikit-learn` (Isolation Forest, aba Sinais Vitais) — importado incondicionalmente no topo de `app.py` junto com todas as outras 3 abas — então o gatilho ("sklearn importado + rerun") está sempre presente em qualquer sessão real da aplicação, não é um cenário de borda raro.

## Goals / Non-Goals

**Goals:**
- Eliminar o segfault ao renderizar qualquer `st.dataframe`/`st.table` da aplicação, em uso real e em teste, em qualquer aba (o gatilho de "sklearn importado" está presente globalmente, então o crash podia acontecer em qualquer aba, não só na que usa o Isolation Forest).
- Fixar a versão do `pyarrow` — a dependência efetivamente responsável pelo bug — para uma versão conhecidamente estável com esta stack, evitando que uma reinstalação futura do `venv/` regrida silenciosamente para uma versão incompatível.

**Non-Goals:**
- Não investiga/reporta o bug upstream ao projeto pyarrow/scikit-learn/streamlit — fora do escopo deste projeto de Tech Challenge (mas a call stack e o repro mínimo ficam documentados aqui para referência futura).
- Não faz pin de `pandas` — testado e confirmado que a versão do pandas não é relevante para este bug (funciona tanto com `2.2.3` quanto com `3.0.3`, desde que `pyarrow<25`).

## Decisions

### D1: Fixar `pyarrow<25` diretamente, não fixar `pandas`
`pyarrow` é a dependência efetivamente causadora do bug (confirmado por teste isolado variando pyarrow com pandas fixo, e variando pandas com pyarrow fixo). Embora `pyarrow` seja transitiva neste projeto (trazida pelo Streamlit, não importada diretamente por nenhum módulo da aplicação), fixá-la explicitamente em `requirements.txt` é necessário porque nenhuma dependência direta deste projeto controla de forma confiável qual versão de `pyarrow` é resolvida — ao contrário da tentativa inicial (fixar `pandas`), que só funcionava por coincidência de uma resolução de dependências específica, e não endereçava a causa raiz real.
**Alternativa tentada e descartada**: fixar `pandas==2.2.3` sem tocar `pyarrow` — testado e confirmado insuficiente: o crash persiste no cenário real do app (upload de vídeo + YOLOv8-pose + clique em "Processar vídeo") mesmo com essa versão do pandas, porque a causa raiz é `pyarrow==25.0.0`, não a versão do pandas.

## Risks / Trade-offs

- **[Trade-off] Fixar `pyarrow` com um limite superior (`<25`) em vez de uma versão exata** → aceito; qualquer versão 24.x é aceitável (o projeto não usa nenhuma API de `pyarrow` diretamente), e um limite superior é suficiente para impedir a regressão para a série 25.x que tem o bug, sem exigir atualização deste pin a cada patch da série 24.x.
- **[Risco] O bug pode ser específico deste ambiente (WSL2, build de pyarrow usada, versão do Streamlit/scikit-learn) e não se manifestar em outro ambiente** → aceito; a correção (pin de versão testada e funcional) não tem custo funcional mesmo se o bug não existisse em outro ambiente.
- **[Risco] Uma futura versão do scikit-learn, Streamlit ou pyarrow pode alterar o comportamento** → aceito; o pin com limite superior (não uma versão exata) permite atualizações de patch dentro da série 24.x sem exigir nova investigação, mas uma atualização deliberada para pyarrow 25+ no futuro deve reproduzir este cenário de teste antes de ser aceita.

## Migration Plan

Quem já tinha o `venv/` criado antes desta change precisa reinstalar as dependências:
```bash
source venv/bin/activate
pip install -r requirements.txt
```
Isso resolve `pyarrow` para a série 24.x, substituindo a versão 25.x previamente instalada. Não força reinstalação do `pandas` (qualquer versão `>=2.2.0` já instalada continua válida).

## Open Questions

Nenhuma pendente.
