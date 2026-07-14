## 1. Cálculo da sugestão de sensibilidade

- [x] 1.1 Implementar `video/analysis.py::suggest_sensitivity_threshold(frame_series, window=DEFAULT_WINDOW) -> float` — heurística simples baseada na variação (desvio padrão) das séries de ângulo/velocidade do vídeo; retorna o valor de sensibilidade padrão atual (`2.0`) quando não há nenhum frame com pose detectada (série vazia/toda NaN)
- [x] 1.2 Escrever testes cobrindo: vídeo com movimento estável (sugestão dentro do intervalo do slider), vídeo com bastante variação (sugestão mais alta), vídeo sem nenhuma pose detectada (fallback para o valor padrão), série muito curta (menor que a janela)

## 2. Integração na aba Vídeo

- [x] 2.1 Mover a chamada de `_extract_pose_frame_series` em `app.py` para rodar automaticamente após um upload válido (fora do bloco do botão "Processar vídeo"), reaproveitando o cache já existente
- [x] 2.2 Calcular a sugestão via `suggest_sensitivity_threshold` sobre o resultado dessa extração e usá-la como `value=` inicial do slider de sensibilidade
- [x] 2.3 Exibir uma mensagem (`st.caption`) próxima ao slider indicando que aquele é um valor sugerido para o vídeo carregado, deixando claro que pode ser ajustado livremente
- [x] 2.4 Garantir que clicar "Processar vídeo" continua funcionando (reaproveitando a extração já feita no upload, sem repetir o processamento pesado) e que os alertas/relatório de desvios seguem corretos
- [x] 2.5 Tratar o caso de erro no carregamento do modelo/extração de pose no momento do upload (antes do clique) com a mesma mensagem de erro amigável já usada no fluxo atual, sem quebrar a aba

## 3. Verificação

- [x] 3.1 Rodar suíte completa de testes (`venv/bin/python -m pytest tests/ -q`) — 107/107 passando (5 novos testes de `suggest_sensitivity_threshold`)
- [x] 3.2 Confirmar boot do app (`streamlit run app.py`) sem erros, e validar via `AppTest` que o slider aparece pré-populado após o upload de um vídeo real — confirmado com `data/demo_pose_walk.mp4` real (sem mock de modelo): slider abriu em `1.66` em vez do fixo `2.0` anterior, sem exceção. (Clicar "Processar vídeo" depois segfaulta no `AppTest` por causa da limitação conhecida `AppTest`+`st.dataframe`/pandas/pyarrow já documentada no README — não é regressão desta change, o boot real via `streamlit run` funciona normalmente.)
- [x] 3.3 Atualizar `README.md` (descrição da aba Vídeo + contagem de testes)

## 4. Fixes do code review

- [x] 4.1 **Important**: dois dos quatro testes novos não exercitavam de fato o comportamento que diziam testar — `test_suggest_sensitivity_threshold_is_within_slider_bounds_for_stable_motion` usava série exatamente constante (std=0), caindo silenciosamente no caminho de fallback em vez do cálculo real por quantil; `test_suggest_sensitivity_threshold_is_higher_for_more_variable_motion` comparava osciladores periódicos que são invariantes a escala sob z-score, produzindo valores idênticos e só passando por usar `>=` em vez de `>`. Corrigido: ambos os testes agora usam um padrão de jitter pequeno e não-nulo (evita o fallback) e o teste de "mais variável" usa outliers reais (não apenas amplitude maior de um padrão periódico), com `assert suggestion != FALLBACK_SENSITIVITY_THRESHOLD` e `assert variable_suggestion > stable_suggestion` (desigualdade estrita)
- [x] 4.2 **Important**: `video/analysis.py::_rolling_zscore` duplicava a fórmula de `anomaly/zscore.py::detect_anomalies` (mesmo cálculo de z-score, risco de as duas implementações divergirem silenciosamente no futuro). Corrigido: extraído `anomaly/zscore.py::rolling_zscore(series, window) -> pd.Series` (retorna a magnitude bruta, não booleana) como função pública compartilhada; `detect_anomalies` e `video.analysis.suggest_sensitivity_threshold` agora chamam a mesma implementação
- Minor findings do review (caption idêntico entre sugestão real e fallback; mensagem de erro genérica ao clicar "Processar vídeo" após falha de carregamento do modelo; docstring desatualizada sobre múltiplos call sites) não foram endereçados — são polish de UX/documentação sem risco funcional, deixados para uma iteração futura se necessário
