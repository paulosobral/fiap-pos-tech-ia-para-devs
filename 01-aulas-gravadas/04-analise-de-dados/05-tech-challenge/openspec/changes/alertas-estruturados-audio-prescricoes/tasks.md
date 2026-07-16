## 1. Áudio — `alert_id` encadeado entre as 3 fontes

- [x] 1.1 `audio/aws_speech.py::raise_critical_term_alerts`: aceitar `start_index: int = 1`, atribuir `alert_id=f"#A{n:02d}"` sequencialmente a cada alerta gerado (ordem de `find_critical_terms`), retornando também o próximo índice livre (ou deixando o chamador calcular `start_index + len(alerts)`)
- [x] 1.2 `audio/analysis.py::analyze`: aceitar `start_index: int = 1`, atribuir `alert_id=f"#A{n:02d}"` sequencialmente na ordem já existente (taxa de fala, depois pausas)
- [x] 1.3 `app.py` (aba Áudio): encadear os índices — `raise_critical_term_alerts(...)` com o default, depois `analyze(..., start_index=len(critical_alerts) + 1)`
- [x] 1.4 `app.py` (aba Áudio): exibir o `alert_id` junto de cada item — termo crítico, segmento de taxa de fala anômala, pausa anômala — no mesmo formato usado em Sinais Vitais/Vídeo (prefixo do id antes do texto)
- [x] 1.5 Testes (`tests/test_audio_aws_speech.py`, `tests/test_audio_analysis.py`): ids sequenciais corretos com `start_index` default e customizado; `alert_id` presente e único em cada alerta gerado

## 2. Prescrições — `alert_id` por finding

- [x] 2.1 `prescriptions/bedrock_review.py::generate_alerts_for_findings`: atribuir `alert_id=f"#P{n:02d}"` na ordem de iteração de `findings`
- [x] 2.2 `app.py` (aba Prescrições): exibir o `alert_id` junto do card de cada inconsistência (iterar `findings` e `review_result["alerts"]` pareados)
- [x] 2.3 Testes (`tests/test_bedrock_review.py`): `alert_id` sequencial e único por finding; card e alerta compartilham o mesmo id

## 3. Dataset sintético de prescrições sem anomalias

- [x] 3.1 Criar `scripts/gen_prescriptions_demo.py`: gera `data/prescricoes_sinteticas_normal.csv` para os mesmos pacientes (A/B/C) do dataset atual, com dose constante por medicamento e sem a combinação de risco conhecida (Warfarina+Aspirina) usada no prompt de revisão; documentar no docstring que a garantia é sobre a construção do dataset, não sobre a resposta do Bedrock
- [x] 3.2 Gerar `data/prescricoes_sinteticas_normal.csv` executando o script
- [x] 3.3 Validar manualmente que o arquivo carrega sem erro pela aba Prescrições (colunas corretas, `load_prescriptions` aceita)

## 4. Verificação e documentação

- [x] 4.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde
- [x] 4.2 Validar via boot: processar `data/demo_consulta_audio.mp3` (ou fixture) na aba Áudio e confirmar ids `#A01`, `#A02`... visíveis junto dos itens; revisar `data/prescricoes_sinteticas_normal.csv` na aba Prescrições e confirmar carregamento e (se houver algum finding) id `#P01`... visível
- [x] 4.3 Atualizar `README.md`/`RELATORIO_TECNICO.md` (Áudio e Prescrições também vinculam alerta↔item de origem; novo dataset de prescrições sem anomalias)
