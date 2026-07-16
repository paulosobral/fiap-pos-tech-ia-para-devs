## Context

A aba Sinais Vitais (`app.py`, bloco `with tab_vitals:`) hoje: upload de CSV → gráfico de linha → dois `st.number_input` ("Janela do rolling z-score", "Threshold do z-score (|z| >)") → aviso de combinação inefetiva (já existe, da change do bugfix) → botão "Processar" → `analyze()` → renderiza `combined_report` (um `st.dataframe` com colunas `timestamp`, `zscore_anomaly`, `isolation_forest_anomaly`, os sinais vitais crus, e `agreement` mapeado só para 3 strings) + um loop de `st.warning` com os alertas.

`analyze` retorna `{zscore_anomalies, isolation_forest_anomalies, combined_report, alerts}`. `combined_report` é um DataFrame com uma linha por leitura; coluna `agreement` ∈ {`normal`, `zscore_only`, `isolation_forest_only`, `alta_confianca`}. Colunas de sinal reconhecidas: heart_rate, spo2, resp_rate/respiratory_rate, systolic_bp, diastolic_bp, blood_pressure, temperature.

A aba Vídeo já foi repaginada de forma análoga (linguagem amigável, resumo, helpers de apresentação puros em `video/analysis.py` como `group_events_for_display`/`joint_label`/`JOINT_LABELS`). Este design segue o mesmo padrão para Sinais Vitais.

Decisões fechadas no brainstorming: nomes amigáveis + termo técnico no tooltip; controles renomeados com tooltip+exemplo; resumo em linguagem clara + tabela com cabeçalhos amigáveis e valores traduzidos; 3 níveis de confiança com frase clara + ícone/cor + tooltip.

## Goals / Non-Goals

**Goals:**
- Tornar a saída da aba interpretável por não-técnicos, sem esconder o termo técnico de quem quiser (fica no tooltip).
- Manter `app.py` fino: rótulos, mapa de níveis e montagem do resumo são funções puras testáveis em `vital_signs/analysis.py`.

**Non-Goals:**
- Não alterar a detecção, thresholds, defaults, nem `analyze`/`anomaly`/`isolation_forest`.
- Não mexer nas outras abas nem no feed.
- Não remover a informação dos 3 níveis (o usuário optou por explicá-los, não por filtrar só alta confiança).

## Decisions

### D1: Helpers de apresentação puros em `vital_signs/analysis.py`
Adicionar (colocados junto do vocabulário de apresentação, como no `video/analysis.py`):
- `VITAL_SIGN_LABELS: Dict[str,str]` — heart_rate→"Frequência cardíaca", spo2→"Saturação de O₂ (SpO₂)", resp_rate/respiratory_rate→"Frequência respiratória", systolic_bp→"Pressão sistólica", diastolic_bp→"Pressão diastólica", blood_pressure→"Pressão arterial", temperature→"Temperatura". Helper `vital_sign_label(col)` com fallback para a própria coluna.
- `CONFIDENCE_LEVELS: Dict[str,dict]` — para cada `agreement` não-normal: `{label, icon, short, help}`:
  - `alta_confianca`: label "Alta confiança", ícone 🔴, short "As duas análises concordam — mais provável ser real", help longo.
  - `zscore_only`: label "Só tempo real", ícone 🟠, short "Pico isolado momentâneo", help.
  - `isolation_forest_only`: label "Só histórico", ícone 🟡, short "Fora do padrão geral, sem pico súbito", help.
  Helper `confidence_level(agreement) -> dict`.
- `build_vitals_summary(combined_report) -> dict` — a partir do `combined_report`, produz um resumo estruturado para a UI: total de leituras anômalas, contagem por nível, e uma lista curta de itens legíveis (ex.: `{"nivel": "alta_confianca", "sinal_label": "Frequência cardíaca", "valor": 165.0, "timestamp": ...}`) para as leituras críticas (priorizando alta confiança). Determinístico; a UI formata a frase a partir disso. Documentar quais sinais entram como "o sinal responsável" numa linha (o(s) sinal(is) cujo z-score disparou; se só o Isolation Forest marcou, indicar "padrão geral" em vez de um sinal específico).
**Alternativa considerada**: montar tudo no `app.py` — rejeitada (UI grossa, lógica sem teste), consistente com a escolha feita na aba Vídeo.

### D2: `app.py` — apresentação amigável (só render, sem lógica de dados)
- Camadas: onde hoje o caption cita "rolling z-score" e "Isolation Forest", usar "Detecção em tempo real" e "Análise do histórico completo", com `help=` trazendo o termo técnico e o que faz.
- Controles: `st.number_input` "Sensibilidade" (era threshold) e "Tamanho da janela de comparação" (era window), cada um com `help=` explicando o efeito em linguagem simples com exemplo (padrão da aba Vídeo). Manter `zscore_threshold_is_reachable` + o aviso.
- Resultado:
  - Resumo no topo via `build_vitals_summary` — uma frase/bullets ("N leituras críticas — {sinal} {alta/baixa} às {hora}...").
  - Tabela: partir do `combined_report` filtrado (não-normal), montar um DataFrame de exibição com cabeçalhos "Horário", "Sinal vital", "Valor", "Nível de confiança" (com ícone+label via `confidence_level`), traduzindo nomes de sinal via `vital_sign_label`. (Uma linha por leitura anômala; o sinal exibido é o responsável pela marcação — se múltiplos, listar; se só histórico, "padrão geral".)
  - Legenda dos 3 níveis: um pequeno bloco explicativo (ícone + short) para o usuário decodificar a coluna "Nível de confiança", com o help longo em tooltip.
- Alertas: manter o bloco do feed compartilhado (só garantir que a descrição já é legível — não é foco desta change mudar o texto do alerta).

### D3: Não quebrar o `st.dataframe` (segfault conhecido)
A aba usa `st.dataframe`. O projeto tem `pyarrow<25` pinado justamente por causa do segfault com sklearn+dataframe; esta change não muda isso. A validação via `AppTest` pode esbarrar no problema conhecido — nesse caso, validar via boot real + teste direto dos helpers (mesmo procedimento das changes anteriores).

## Risks / Trade-offs

- **[Trade-off] Traduzir/renomear afasta a UI dos nomes técnicos "auditáveis"** → mitigado: o termo técnico fica no tooltip e no relatório técnico; nada é escondido, só reposicionado.
- **[Risco] "Sinal responsável" por uma linha anômala pode ser ambíguo quando vários sinais disparam no mesmo timestamp** → tratado em `build_vitals_summary`/montagem da tabela listando os sinais marcados naquela linha; se só o Isolation Forest marcou (sem z-score de um sinal específico), rotular como "padrão geral".
- **[Trade-off] Mais texto explicativo na tela** → aceito; é o objetivo (interpretabilidade), e a legenda dos níveis é compacta (ícone + frase curta, detalhe no tooltip).

## Migration Plan

Não aplicável — mudança de apresentação de UI, sem dados/persistência.

## Open Questions

Nenhuma — nomes das camadas, controles, formato do resultado e explicação dos níveis fechados no brainstorming.
