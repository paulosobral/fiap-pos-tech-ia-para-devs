## ADDED Requirements

### Requirement: Versão do pyarrow compatível com a renderização de tabelas do Streamlit
O sistema SHALL fixar um limite superior de versão para `pyarrow` em `requirements.txt` (dependência transitiva do Streamlit/pandas usada na serialização Arrow de `st.dataframe`/`st.table`), evitando que a resolução automática de dependências (`pip install -r requirements.txt`) selecione uma versão de `pyarrow` que trave o processo.

#### Scenario: Renderização de tabela em qualquer aba não trava o processo
- **WHEN** qualquer aba (Sinais Vitais, Vídeo, Áudio ou Prescrições) renderiza uma tabela via `st.dataframe`/`st.table` durante uso normal da aplicação, incluindo em reruns do Streamlit posteriores ao primeiro (ex.: dentro de um bloco de botão)
- **THEN** a renderização completa sem lançar `segmentation fault` ou qualquer outro crash do processo Python

#### Scenario: Reinstalação do ambiente resolve uma versão de pyarrow dentro do limite fixado
- **WHEN** o `venv/` do projeto é criado do zero e `pip install -r requirements.txt` é executado
- **THEN** a versão do `pyarrow` instalada respeita o limite superior fixado em `requirements.txt`, não uma versão mais recente que o exceda
