## Why

A aplicação Streamlit trava com `segmentation fault` durante uso real (não só em teste automatizado) ao renderizar `st.dataframe` — reproduzido no fluxo real da aba Vídeo (upload → "Processar vídeo"). `requirements.txt` não fixava `pyarrow` (dependência transitiva do Streamlit/pandas), então o `pip install` resolveu para `pyarrow==25.0.0`, versão que tem um bug real de segfault dentro de `libarrow.so.2500` (em `pyarrow/pandas_compat.py::convert_column`) ao serializar um `DataFrame` para Arrow — mas apenas quando `scikit-learn` está importado no processo (o app importa incondicionalmente, para a aba Sinais Vitais) **e** a renderização ocorre num script-rerun do Streamlit posterior ao primeiro (ex.: dentro de `if st.button(...):`). Como esse gatilho está sempre presente em qualquer sessão real da aplicação, o crash não é um caso de borda raro — acontece de forma consistente e reproduzível (confirmado 100% das vezes num repro mínimo isolado, e no fluxo real do app). Já era uma limitação conhecida documentada no README/RELATORIO_TECNICO (só observada até então via `AppTest`); o relato do usuário confirma que também ocorre em uso real via `streamlit run`.

Uma hipótese inicial (fixar a versão do `pandas`) foi testada e descartada por ser insuficiente — não resolvia o cenário real completo do app. A causa raiz isolada corretamente é a versão do `pyarrow`.

## What Changes

- Fixa `pyarrow<25` em `requirements.txt` (limite superior explícito na dependência transitiva efetivamente responsável pelo bug), forçando uma versão compatível com Streamlit + scikit-learn nesta stack.
- Reinstala o `venv/` do projeto com a versão corrigida.
- Atualiza README/RELATORIO_TECNICO removendo a menção à limitação conhecida de segfault (o problema deixa de existir com o pin) e adiciona uma nota explicando por que `pyarrow` tem um limite superior fixado neste projeto, incluindo a causa raiz (conflito pyarrow 25.x + scikit-learn + rerun do Streamlit), para não regredir silenciosamente numa reinstalação futura.
- **BREAKING** (ambiente de desenvolvimento, não código do usuário): quem já tinha `venv/` instalado com `pyarrow==25.0.0` precisa reinstalar as dependências (`pip install -r requirements.txt`) para pegar o fix.

## Capabilities

### New Capabilities
- `project-dependencies`: garante que a versão fixada de `pyarrow` (dependência transitiva do Streamlit/pandas) não causa falha na renderização de tabelas (`st.dataframe`/`st.table`) usadas pelas 4 abas.

### Modified Capabilities
(nenhuma — não altera comportamento de nenhuma capability funcional existente, apenas fixa uma versão de dependência transitiva)

## Impact

- **Dependências**: `requirements.txt` (`pyarrow<25` fixado; `pandas` permanece `>=2.2.0`, sem alteração — testado e confirmado que a versão do pandas não é relevante para este bug).
- **Ambiente**: reinstalação do `venv/` necessária.
- **Documentação**: README.md, RELATORIO_TECNICO.md (remoção da seção de limitação conhecida, já que o crash deixa de ocorrer).
- **Sem impacto** em código de nenhuma capability (`anomaly/`, `video/`, `audio/`, `vital_signs/`, `prescriptions/`, `alerts/`) — nenhuma dessas usa API de `pyarrow` diretamente.
