## Context

`Alert` já suporta `alert_id`/`category`/`level` opcionais (change `alertas-estruturados-e-vinculo-sinais-vitais`). Vídeo atribui `#V01`, `#V02`... a eventos ordenados cronologicamente (frame time); Sinais Vitais atribui `#S01`, `#S02`... por linha anômala (ordem das linhas). Em ambos, o id é calculado numa única passada determinística e usado tanto no alerta quanto no elemento correspondente exibido na própria aba (frame da galeria / linha da tabela).

Áudio e Prescrições ainda não atribuem `alert_id`:
- Áudio tem 3 fontes de alerta chamadas em sequência fixa por `app.py` (`raise_critical_term_alerts` → depois, dentro de `analyze`, taxa de fala → pausa), sem uma linha do tempo única compartilhada entre elas (termos críticos são achados por posição no texto, não por tempo de fala).
- Prescrições tem 1 fonte (`generate_alerts_for_findings`), chamada uma vez por revisão, iterando a lista de `findings` retornada pelo Bedrock em ordem.

## Goals / Non-Goals

**Goals:**
- Áudio: todo alerta (termo crítico, taxa de fala anômala, pausa anômala) carrega um `alert_id` único (`#A01`, `#A02`, ...), na MESMA ordem em que os alertas já aparecem na aba (termos críticos primeiro, depois taxa de fala, depois pausas — ordem atual de `app.py`), e a UI mostra esse id junto de cada item correspondente.
- Prescrições: todo alerta de inconsistência carrega um `alert_id` único (`#P01`, `#P02`, ...), na ordem dos `findings` retornados pelo Bedrock, e a UI mostra esse id junto do card da inconsistência correspondente.
- Novo dataset `data/prescricoes_sinteticas_normal.csv`, gerado por script reproduzível, sem nenhuma inconsistência real esperada (embora a decisão final seja do Bedrock, o dataset é construído para não conter mudança abrupta, interação de risco nem dose sem justificativa).

**Non-Goals:**
- Não muda a detecção, thresholds ou o texto das descrições já exibidas — só adiciona o campo estruturado e o prefixo visual do id, como nas changes anteriores.
- Não introduz uma linha do tempo unificada entre termos críticos (baseados em texto) e taxa de fala/pausa (baseados em tempo) — o id é só uma sequência de exibição, não implica ordem temporal cruzada.
- Não garante deterministicamente "zero findings" no dataset normal — o Bedrock é um LLM; a garantia é sobre a CONSTRUÇÃO do dataset (sem os padrões que o prompt pede para detectar), não sobre a resposta do modelo em cada execução.

## Decisions

### D1: Áudio — contador de id encadeado entre as 3 fontes via parâmetro `start_index`
`raise_critical_term_alerts(text, terms=..., context_chars=..., start_index=1)` e `analyze(words, window=..., threshold=..., segment_seconds=..., start_index=1)` passam a aceitar `start_index` (default 1) e atribuir `alert_id=f"#A{n:02d}"` sequencialmente a cada alerta que geram (termos críticos na ordem de `find_critical_terms`; dentro de `analyze`, taxa de fala primeiro, depois pausa — ordem já existente no laço). Cada função continua pura/testável isoladamente (id parte de 1 por padrão). `app.py` encadeia explicitamente:
```python
critical_alerts = raise_critical_term_alerts(text, terms=critical_terms)
...
audio_result = analyze(words, window=..., threshold=..., start_index=len(critical_alerts) + 1)
```
Resultado: a sequência de ids na sidebar/feed acompanha a ordem em que o usuário já vê os alertas na aba (topo → base), sem precisar de um contador global em `st.session_state` nem de um timestamp comum entre as 3 fontes.
**Alternativa considerada**: um contador global em `session_state` — rejeitada; acopla os módulos de análise ao Streamlit e complica os testes (que hoje chamam as funções isoladamente sem sessão simulada além do `alerts`).
**Alternativa considerada**: prefixos por sub-fonte (`#AT01` termos, `#AR01` taxa, `#AP01` pausa) — rejeitada; o padrão já estabelecido (Vídeo, Sinais Vitais) é um único contador por aba, e um único contador é mais simples de explicar ao usuário.

### D2: Prescrições — id por posição na lista de findings
`generate_alerts_for_findings(findings, patient_name)` passa a atribuir `alert_id=f"#P{n:02d}"` na ordem em que itera `findings` (ordem já determinística: a ordem retornada pelo Bedrock). Sem parâmetro extra — só uma fonte, uma chamada por revisão.

### D3: UI — exibir o id junto ao item de origem
Em `app.py`, nos 3 laços que hoje fazem `st.warning(f"[{ts}] {desc}")` para os alertas de Áudio, prefixar com o id (mesmo padrão já usado em Sinais Vitais/Vídeo): `id_prefix = f"{alert.alert_id} " if alert.alert_id else ""`. No card de inconsistência de Prescrições (hoje iterado sobre `findings`, não sobre `alerts`), passar a iterar `zip(findings, review_result["alerts"])` (mesma ordem, um alerta por finding — já garantido por D2) para mostrar o id junto de cada card.

### D4: Dataset `prescricoes_sinteticas_normal.csv` — script gerador análogo ao de Sinais Vitais
Novo `scripts/gen_prescriptions_demo.py`: constrói, para os mesmos 3 pacientes (A/B/C) do arquivo atual, um histórico com dose CONSTANTE em cada medicamento e sem combinação presente na lista de risco conhecida usada no prompt (Warfarina+Aspirina), preservando o formato (`paciente,medicamento,dose,data`). Documenta no docstring que a garantia é sobre a construção do dataset, não sobre a resposta do Bedrock (não dá para "verificar automaticamente" como no z-score, pois depende de uma chamada real ao LLM — custaria uma chamada AWS por execução do gerador). `data/prescricoes_sinteticas.csv` (atual) passa a ser documentado como o par "com anomalias".
**Alternativa considerada**: rodar o Bedrock dentro do script gerador para autoverificar "zero findings", como o script de sinais vitais faz com `analyze` — rejeitada; sinais vitais verifica um modelo estatístico local determinístico, enquanto aqui seria uma chamada de rede real e não-determinística (custo e flakiness), fora do orçamento do projeto.

## Risks / Trade-offs

- **[Risco] `start_index` em `analyze`/`raise_critical_term_alerts` muda a assinatura pública** → mitigado por ter default `1`, retrocompatível com todo teste/chamador existente que não passa o argumento.
- **[Trade-off] Dataset normal não garante "zero findings" do Bedrock** → aceito; documentado no docstring do gerador — mesma limitação já reconhecida em `RELATORIO_TECNICO.md` sobre o dataset pequeno/sintético de prescrições.

## Migration Plan

Não aplicável — extensão retrocompatível (parâmetros com default, novo arquivo de dados aditivo).

## Open Questions

Nenhuma.
