## 1. Corrigir o default e sinalizar combinações inefetivas

- [x] 1.1 Alterar `DEFAULT_WINDOW` de `6` para `13` em `vital_signs/analysis.py` (teto `sqrt(12)≈3.46 > DEFAULT_THRESHOLD=3.0`); manter `DEFAULT_THRESHOLD=3.0`
- [x] 1.2 Adicionar helper puro em `vital_signs/analysis.py`: `zscore_threshold_is_reachable(window, threshold) -> bool` (True quando `threshold < sqrt(window-1)`)
- [x] 1.3 Na aba Sinais Vitais (`app.py`): após ler os `number_input` de janela e threshold, se `not zscore_threshold_is_reachable(...)`, exibir `st.warning` explicando que nenhuma anomalia z-score é detectável (|z| máx = `sqrt(janela-1)`) e sugerindo aumentar a janela ou baixar o threshold

## 2. Testes

- [x] 2.1 Teste: com `DEFAULT_WINDOW`/`DEFAULT_THRESHOLD`, uma série com um pico claro é marcada como anômala pela camada z-score (antes: zero) — prova que o default é efetivo
- [x] 2.2 Teste: `zscore_threshold_is_reachable` — True para (13, 3.0), False para (6, 3.0), fronteira coerente (ex.: (10, 3.0) onde `sqrt(9)=3.0` não é `> 3.0` → False)
- [x] 2.3 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde; conferir que nenhum teste existente dependia de `DEFAULT_WINDOW=6`

## 3. Verificação e documentação

- [x] 3.1 Validar via `AppTest`/boot que a aba Sinais Vitais processa com o novo default sem erro, e que a combinação inefetiva dispara o aviso
- [x] 3.2 Atualizar `README.md`/`RELATORIO_TECNICO.md` onde mencionarem os parâmetros do z-score de sinais vitais (janela padrão 13; nota sobre o teto `sqrt(janela-1)` e o aviso da UI)
