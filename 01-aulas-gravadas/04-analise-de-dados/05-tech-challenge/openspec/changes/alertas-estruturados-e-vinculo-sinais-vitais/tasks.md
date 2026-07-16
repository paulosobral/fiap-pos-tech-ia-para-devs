## 1. Estender o modelo de alerta (`alerts/feed.py`)

- [x] 1.1 Adicionar a `Alert` os campos opcionais `alert_id: Optional[str] = None`, `category: Optional[str] = None`, `level: Optional[str] = None` (após os campos atuais, preservando os posicionais)
- [x] 1.2 Estender `add_alert(origin, description, timestamp=None, alert_id=None, category=None, level=None)` repassando os novos campos ao `Alert`
- [x] 1.3 Testes: `add_alert` sem os novos args cria alerta com eles `None` (retrocompat); com os args, preenche corretamente; `get_alerts` continua newest-first

## 2. Sinais Vitais: ID de vínculo + category/level + classificação visual

- [x] 2.1 Em `vital_signs/analysis.py::analyze`, atribuir `#S01`, `#S02`, ... determinístico a cada linha anômala exibida (z-score e/ou Isolation Forest); expor o ID por linha no resultado/`combined_report` para a tabela
- [x] 2.2 Ao gerar os alertas de z-score, passar `alert_id` (o ID da linha), `category=vital_sign_label(sinal)` e `level` (o `agreement` da linha) ao `add_alert`; múltiplos sinais na mesma linha compartilham o ID da linha
- [x] 2.3 Em `app.py` (aba Sinais Vitais): exibir a coluna de ID na tabela de anomalias (mesmo valor que no alerta); no bloco inline "Alertas gerados", usar o ícone/cor do nível (`confidence_level`) em vez de `st.warning` uniforme
- [x] 2.4 Testes: IDs únicos/determinísticos por linha anômala; alerta de sinais vitais carrega alert_id/category/level coerentes com a linha; a tabela expõe o mesmo ID

## 3. Migrar Vídeo/Áudio/Prescrições para campos estruturados

- [x] 3.1 Vídeo (`video/analysis.py`): passar `alert_id=event["event_id"]`, `category=event_category(event)` e um `level` sensato ao `add_alert`, mantendo o texto atual da descrição
- [x] 3.2 Áudio (`audio/*`): preencher `category` (ex.: "Termo crítico" / "Fadiga de fala") e `alert_id` se houver identificador natural (senão `None`), mantendo o texto
- [x] 3.3 Prescrições (`prescriptions/bedrock_review.py`): preencher `category` (ex.: "Inconsistência de prescrição") e `alert_id` se aplicável, mantendo o texto
- [x] 3.4 Testes: cada uma dessas abas preenche os campos estruturados de forma coerente com o texto (sem alterar o texto que o usuário já vê)

## 4. Verificação e documentação

- [x] 4.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde (as 4 abas)
- [x] 4.2 Validar via `AppTest`/boot: Sinais Vitais mostra ID na tabela e nos alertas com ícone de nível; Vídeo continua com seu #V no alerta; nenhuma aba quebra. Se o segfault conhecido `AppTest`+`st.dataframe` atrapalhar, validar via boot + teste direto
- [x] 4.3 Atualizar `README.md`/`RELATORIO_TECNICO.md` (alertas com ID de vínculo em Sinais Vitais + classificação visual por nível; nota de que Alert agora carrega id/categoria/nível estruturados)
