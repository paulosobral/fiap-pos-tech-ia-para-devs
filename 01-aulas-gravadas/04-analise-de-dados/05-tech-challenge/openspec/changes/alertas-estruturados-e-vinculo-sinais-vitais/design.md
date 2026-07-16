## Context

`alerts/feed.py` define `Alert{origin, description, timestamp}` e `add_alert(origin, description, timestamp=None)`; os alertas vivem em `st.session_state["alerts"]` e `get_alerts()` retorna newest-first. As 4 abas chamam `add_alert` com origin+description (texto já formatado).

- Vídeo (`video/analysis.py`): já atribui `event_id` (`#V01`) e categoria (Cabeça/Braços/...) por evento, e embute ambos no TEXTO do alerta (`_event_description`: `"#V03 [Braços] ..."`). Os campos existem no dict do evento, mas o alerta só recebe o texto.
- Sinais Vitais (`vital_signs/analysis.py`): gera um alerta por leitura anômala do z-score (texto "Leitura anômala de {col} = {val} em {ts} ..."), SEM ID e sem nível estruturado. A UI (`app.py`) mostra a tabela `combined_report` (com `agreement`/`confidence_level`) e os alertas como `st.warning`.
- Áudio (`audio/*`) e Prescrições (`prescriptions/bedrock_review.py`): geram alertas com texto próprio, sem campos estruturados.

Esta change torna ID/categoria/nível dados de PRIMEIRA CLASSE no `Alert` (não só texto), adiciona o vínculo alerta↔linha em Sinais Vitais, e melhora a classificação visual dos alertas de sinais vitais no feed. É a base para o export unificado (change B).

## Goals / Non-Goals

**Goals:**
- `Alert` carrega `alert_id`, `category`, `level` opcionais, retrocompatível.
- Sinais Vitais: ID único por leitura anômala, no alerta e na tabela; alerta com category (sinal) e level (confiança); feed com ícone/cor por nível.
- Vídeo/Áudio/Prescrições preenchem os campos estruturados (mantendo o texto atual da descrição).

**Non-Goals:**
- Não implementar o export ainda (change B).
- Não mudar a detecção, thresholds, nem o texto legível que o usuário já vê (só ADICIONAR os campos estruturados e o indicador visual).
- Não reordenar o feed nem mudar `get_alerts()`.

## Decisions

### D1: Campos opcionais no `Alert` + `add_alert`
```
@dataclass
class Alert:
    origin: str
    description: str
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: Optional[str] = None
    category: Optional[str] = None
    level: Optional[str] = None
```
`add_alert(origin, description, timestamp=None, alert_id=None, category=None, level=None)`. Todos os chamadores atuais continuam válidos (novos args têm default `None`). Ordem dos campos preserva os posicionais existentes.
**Alternativa considerada**: um dict `metadata` genérico — rejeitada; campos nomeados são mais claros, testáveis e diretos pro export em colunas.

### D2: ID de Sinais Vitais (`#S01`) — vínculo alerta↔linha
Prefixo `#S` (distinto de `#V` do vídeo — coerente com "Sinais"). Em `vital_signs/analysis.py::analyze`, ao percorrer as leituras anômalas do z-score (que já geram alertas), atribuir sequencialmente `#S01`, `#S02`, ... em ordem determinística (ordem das linhas anômalas). O ID é:
- passado a `add_alert(..., alert_id=id, category=vital_sign_label(sinal), level=agreement_da_linha)`;
- exposto no `combined_report`/no resultado para a UI, de modo que a tabela possa exibir uma coluna "ID" com o mesmo valor na linha correspondente.
Como o alerta de z-score é por (linha, sinal) e a tabela é por linha, definir a regra: o ID identifica a LEITURA anômala (linha); se uma linha dispara em múltiplos sinais, o ID é da linha e o alerta de cada sinal referencia o mesmo ID da linha (documentar). A UI mostra o ID na linha da tabela e no alerta — casáveis. Nota: só linhas que geram alerta (z-score) recebem ID; linhas só-Isolation-Forest aparecem na tabela mas podem não ter alerta no feed (comportamento atual preservado); para essas, exibir ID na tabela mesmo assim é opcional — decisão: atribuir ID a toda linha anômala exibida na tabela (z-score e/ou IF), e o alerta (quando houver) usa o ID da sua linha. Isso mantém tabela e feed consistentes.
**Alternativa considerada**: ID por (linha,sinal) — rejeitada; multiplicaria IDs e dificultaria o casamento visual com a tabela, que é por linha.

### D3: Classificação visual no feed inline de Sinais Vitais
Onde `app.py` hoje faz `st.warning(f"[{ts}] {desc}")` para cada alerta de sinais vitais, passar a usar o `confidence_level(level)` (ícone + label) já existente: renderizar com o ícone do nível e, se fizer sentido, `st.error` para alta confiança / `st.warning` para os demais — mantendo o texto. Consistente com a tabela e a legenda. (O feed unificado da sidebar é tratado de forma genérica para todas as abas; aqui o foco é o bloco inline "Alertas gerados" da própria aba Sinais Vitais. Se o indicador de nível também couber na sidebar de forma genérica via o novo campo `level`, aplicar lá também — mas sem quebrar as abas que não setam level.)

### D4: Migrar Vídeo/Áudio/Prescrições para preencher os campos estruturados
- Vídeo: já tem event_id e categoria no dict do evento; passar `alert_id=event["event_id"]`, `category=event_category(event)`, `level` (ex.: tipo do evento ou nível) ao `add_alert`, mantendo o texto atual.
- Áudio/Prescrições: preencher pelo menos `category` (ex.: "Termo crítico"/"Fadiga" no áudio; "Inconsistência de prescrição" no prescrições) e `alert_id` se já houver um identificador natural; onde não houver ID, deixar `alert_id=None` (o export lida com ausência). Manter o texto.
Objetivo: o export (change B) lê colunas estruturadas em vez de parsear texto.

## Risks / Trade-offs

- **[Risco] Mexer no `Alert`/`add_alert` afeta as 4 abas + o feed compartilhado** → mitigado por serem campos OPCIONAIS com default None; nenhum chamador atual quebra; re-rodar a suíte completa cobre as 4 abas.
- **[Trade-off] Duplicação de informação (ID/categoria tanto no texto quanto nos campos)** → aceito temporariamente; o texto continua legível para o feed atual, e os campos habilitam o export limpo. Poderíamos parar de embutir no texto, mas isso mudaria o que o usuário já vê — fora de escopo.
- **[Risco] Regra de ID por-linha vs alerta por (linha,sinal) em sinais vitais** → resolvida explicitamente em D2 (ID é da linha; múltiplos sinais na mesma linha compartilham o ID da linha); documentada e testada.

## Migration Plan

Não aplicável — extensão retrocompatível de estrutura em memória, sem persistência.

## Open Questions

Nenhuma — eixo (Alert estruturado) e divisão em 2 changes confirmados com o usuário.
