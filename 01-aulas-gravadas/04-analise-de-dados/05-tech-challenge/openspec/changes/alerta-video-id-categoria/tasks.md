## 1. ID e categoria no evento (`video/analysis.py`)

- [x] 1.1 Atribuir `event_id` (`#V01`, `#V02`, ...) a cada evento em `analyze`, de forma determinística, na ordem cronológica final dos eventos, ANTES de gerar os alertas — gravado no dict do evento (schema ganha `event_id`)
- [x] 1.2 Adicionar `JOINT_CATEGORY` (pescoço→Cabeça; cotovelos→Braços; quadris→Tronco; joelhos→Pernas) e `event_category(event)` (velocidade→Corpo; zona_critica→Zona de risco), junto de `JOINT_LABELS`
- [x] 1.3 Atualizar `_event_description` para o formato `"{event_id} [{categoria}] {texto_atual}"`
- [x] 1.4 Testes: `event_id` presente em todos os eventos, único, determinístico e sequencial; `event_category` mapeia cada articulação/tipo corretamente; `_event_description`/alerta contém ID + categoria + intervalo

## 2. Galeria mostra o ID (`app.py`, aba Vídeo)

- [x] 2.1 Incluir `event_id` na legenda de cada foto da galeria (ex.: "#V03 — 5.2s a 6.4s"), lendo o mesmo campo do evento
- [x] 2.2 Preservar todo o resto (agrupamento por articulação, expanders, top-N, grade, gate do botão, cache, outras abas/feed intactos)

## 3. Verificação e documentação

- [x] 3.1 Rodar a suíte completa (`venv/bin/python -m pytest tests/ -q`) — tudo verde
- [x] 3.2 Validar via `AppTest` (upload `data/demo_pose_walk.mp4` → "Processar vídeo") sem exceção; confirmar que os alertas trazem ID+categoria e que as legendas da galeria trazem o mesmo ID; boot real via `streamlit run app.py` sem traceback
- [x] 3.3 Atualizar `README.md` e `RELATORIO_TECNICO.md` (descrição do alerta de vídeo: ID único + categoria; legenda da galeria com o mesmo ID)
