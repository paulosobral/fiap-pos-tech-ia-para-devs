# Design: SDR Consultivo — gate de agendamento, router rico, memória comercial

**Data:** 2026-09-23  
**Status:** aprovado (design em chat) — aguarda revisão do spec  
**Classificação:** arquitetural (muda estados, enum do router, assinatura do reply_generator)

## Problema

O fluxo trata "lead qualificado" como sinônimo de "lead pronto para visita". Resultado clássico de SDR IA:

```
oi → quer agendar? → não → quer agendar? → não
```

Gargalos:

1. Estados lineares (`intent → qualification → recommendation → scheduling`) não cobrem ida e volta real (pergunta sobre imóvel → refine → compare → gostei).
2. `VALID_ACTIONS` pequeno demais: `refine_search`, `compare_properties`, `visit_interest` caem em `provide_info` e o bot fica perdido.
3. Sem memória comercial: `favorite_property`, `rejected_properties`, `interests` não existem — "essa tem estacionamento?" vira "qual imóvel?".
4. `reply_generator` recebe pouco contexto e não sabe o estágio da conversa.

## Objetivo

Tornar o bot um consultor que **conversa sobre imóveis antes de agendar**, com memória de preferências e gate explícito de prontidão para visita.

## Escopo

4 etapas fatiadas, cada uma com testes verdes antes da próxima.

---

## Etapa 1 — Gate `ready_for_scheduling`

**Arquivos:** `sales_flow.py`, testes (`test_sales_flow.py`, `test_prd_gaps.py`, `test_restriction.py`).

### Comportamento

Novo método em `SalesFlow`:

```python
def ready_for_scheduling(self, state: FlowState) -> bool:
    return bool(
        state.get("favorite_property")
        or state.get("visit_interest")
        or (state.get("lead_info") or {}).get("deadline")
    )
```

- Gate é **OR**: qualquer um dos 3 sinais libera.
- `_node_scheduling` chama o gate **antes** do gate de `shown_count < 3`.
- Se gate falha: volta para `recommendation` com pergunta aberta (nunca "quer agendar?").

### Ordem dos gates em `_node_scheduling`

1. `_wants_options` → `_show_more_options` (inalterado)
2. `ready_for_scheduling` → se False, volta para `recommendation` com pergunta aberta
3. `shown_count < 3 and not lead_id` → mostra mais opções (inalterado)
4. restriction / scheduler (inalterado)

### Ajustes de testes (Etapa 1)

Testes existentes que invocam `current_state: "scheduling"` precisam de um dos sinais no state de entrada:

- `test_sales_flow.py`: `test_restricted_scheduling_defers_action`, `test_unrestricted_scheduling_calls_scheduler`, `test_scheduling_without_checker_runs_normally`, `test_restriction_check_failure_is_fail_open`, `test_restricted_lead_without_lead_id_is_not_restricted` → adicionar `"visit_interest": True` ou `"lead_info": {"deadline": "..."}`.
- `test_prd_gaps.py::test_sales_flow_ics_generated_on_scheduling` → idem.
- `test_restriction.py` (linha ~86) → idem.
- Novos testes: gate bloqueia sem sinal; libera com `favorite_property`; libera com `visit_interest`; libera com `deadline`; resposta de bloqueio é pergunta aberta e `current_state == "recommendation"`.

---

## Etapa 2 — Router rico

**Arquivos:** `llm.py` (`VALID_ACTIONS`, `_ROUTER_SYSTEM_PROMPT`), `sales_flow.py` (`_route_state`, nós), testes (`test_llm.py`, `test_sales_flow.py`).

### Novo enum

```python
VALID_ACTIONS = (
    "provide_info",
    "request_options",
    "refine_search",
    "compare_properties",
    "visit_interest",
    "request_schedule",
    "request_human",
    "decline",
    "unclear",
)
```

### Prompt do router

Adicionar definições com exemplos:

- `refine_search`: quer ajustar critérios ("tem algo menor?", "mais barato?", "outra região?").
- `compare_properties`: quer comparar opções ("qual a diferença entre 1 e 2?", "compara as duas").
- `visit_interest`: demonstra interesse em visitar/imóvel específico ("quero visitar", "gostei da 2", "essa me interessa") **sem** verbo de agendamento explícito.
- `request_schedule` mantém regra de verbo explícito (inalterado).

### Mapeamento em código (nunca LLM escolhe nó)

| action | efeito |
|--------|--------|
| `refine_search` | extrai deltas de `lead_info` → recommendation (refaz RAG) |
| `compare_properties` | recommendation com comparativo dos `properties` em state |
| `visit_interest` | seta `visit_interest=True` (Etapa 3) → scheduling se `ready_for_scheduling` + `shown_count >= 3` |
| `request_schedule` | scheduling (com gates da Etapa 1) |

`_route_state`: `visit_interest` só vai para `scheduling` se `ready_for_scheduling` e `shown_count >= 3`; senão fica no estado atual.

### Ajustes de testes (Etapa 2)

- `test_llm.py`: parser aceita as 3 actions novas; rejeita fora do enum.
- `test_sales_flow.py` (classe `TestAgenticRouter`): mocks de `llm_router` com `refine_search` / `compare_properties` / `visit_interest`.
- `test_handler_llm_secret.py`: sem mudança (só wiring).

---

## Etapa 3 — Memória comercial

**Arquivos:** `sales_flow.py` (`FlowState`, `_node_preprocess`, `_node_recommendation`), `llm.py` (router extrai preferência), testes.

### Schema (`FlowState` + `context` persistido)

Campos em `FlowState` (LangGraph exige no TypedDict):

- `favorite_property: str`
- `rejected_properties: list[str]`
- `visit_interest: bool`
- `interests: list[str]`

Persistência: espelhar em `state["context"]` (já serializado no DynamoDB `Conversation.context`):

```python
context["favorite_property"] = ...
context["rejected_properties"] = [...]
context["visit_interest"] = True
context["interests"] = [...]
```

Seed no `_node_preprocess` a partir de `context` salvo.

### Detecção

1. Router LLM (Etapa 2): além de `lead_info` e `action`, extrai `favorite_property` / `rejected_properties` quando houver ("gostei da Torre Nova", "não quero a 1").
2. Regex baseline (rede de segurança, sem LLM): detecta "gostei da X" / "não quero a X" contra `state["properties"]` (match por número da lista ou substring do title).

### Uso

- `ready_for_scheduling` usa `favorite_property` (Etapa 1).
- Reply_generator recebe `favorite_property` (Etapa 4).
- "Essa tem estacionamento?" → reply contextualizado com o imóvel favorito.

### Ajustes de testes (Etapa 3)

- Seed de `context` → `FlowState`.
- Detecção por regex com `properties` em state.
- Detecção via `llm_router` mock devolvendo `favorite_property`.
- Roundtrip: invoke → `context["favorite_property"]` presente.
- Integração com gate: `favorite_property` libera scheduling.

---

## Etapa 4 — Estado `discovery` + reply_generator rico

**Arquivos:** `sales_flow.py`, `llm.py`, `handler.py`, `entities.py` (`VALID_STATES`), testes.

### Estado `discovery`

- Novo nó no grafo: `_node_discovery`.
- Quando: lead faz pergunta sobre imóvel já mostrado (detalhe: estacionamento, andar, valor) sem pedido de refine nem visit_interest.
- Comportamento: responde com dados do imóvel em foco (`favorite_property` ou último `properties[0]`), **sem** trocar de para recommendation/scheduling.
- Roteamento: `provide_info` + `current_state in (recommendation, discovery)` + mensagem com pergunta sobre detalhe → `discovery`.
- `VALID_STATES` em `entities.py` ganha `"discovery"`; `test_entities.py::test_valid_states_include_all_flow_states` atualizado.

Fluxo alvo:

```
greeting → intent → discovery ⇄ recommendation → visit_interest → scheduling → handoff
```

### `generate_reply` — kwargs ricos

Assinatura atual:

```python
generate_reply(message, canned_response, lead_info, properties, *, api_key, ...)
```

Nova (kwargs com default `None` — backward compat com chamadas posicionais existentes):

```python
generate_reply(
    message, canned_response, lead_info, properties, *,
    api_key,
    favorite_property: str | None = None,
    conversation_stage: str | None = None,
    shown_properties_count: int | None = None,
    model=None, url="", timeout=None, force_complex=False,
) -> str
```

- `_node_postprocess` em `sales_flow.py` passa os kwargs extras.
- `handler.py` `llm_reply` repassa kwargs.
- **Mitigação de testes mockados:** `reply_generator` em testes usa `lambda *_:` ou `**kwargs`; se algum assumir exatamente 4 args, ajustar para `*args, **kwargs`.

### Prompt do reply (substitui `_REPLY_SYSTEM_PROMPT`)

```
Você é um consultor imobiliário corporativo.
NUNCA pareça um formulário.
Faça apenas 1 pergunta por vez.
Se houver imóveis disponíveis: converse sobre eles, explore preferências,
não tente agendar imediatamente.
Só ofereça visita quando: usuário demonstrar interesse explícito
OU mencionar uma opção específica.
Tom: consultivo, profissional, objetivo.
Regras rígidas de segurança (inalteradas):
- só cita imóveis da lista
- nunca inventa preço/metragem/bairro
- máximo 3 frases
- pergunta final coerente com a ação; NUNCA força agendamento
```

Injetar no user block:

```
ESTÁGIO DA CONVERSA: {conversation_stage}
IMÓVEL FAVORITO DO LEAD: {favorite_property or '(nenhum)'}
IMÓVEIS JÁ EXIBIDOS: {shown_properties_count}
```

### Ajustes de testes (Etapa 4)

- `test_llm.py`: `generate_reply` com kwargs novos; prompt contém "consultor" / "1 pergunta"; user block contém estágio e favorito.
- `test_sales_flow.py`: `_node_postprocess` passa kwargs; mocks de `reply_generator` aceitam `**kwargs`.
- `test_entities.py`: `discovery` em `VALID_STATES`.
- Novo: `_node_discovery` responde sem sair do estado; roteamento `provide_info` → discovery.
- `test_handler_llm_secret.py`: `llm_reply` repassa kwargs (smoke).

---

## Fora de escopo

- Mudança de infra Terraform / deploy.
- Novas tabelas DynamoDB (memória vive em `Conversation.context`).
- Mudança no voice-adapter.
- ADR novo de arquitetura de LLM (mantém ADR-011).

## Riscos e mitigações

| Risco | Mitigação |
|-------|-----------|
| Gate quebra testes de scheduling existentes | Etapa 1 ajusta fixtures com `visit_interest`/`deadline`; testes verdes antes da Etapa 2 |
| Assinatura de `generate_reply` quebra mocks | kwargs com default `None`; mocks `*args, **kwargs` |
| `discovery` não reconhecido no DynamoDB | Atualizar `VALID_STATES` + teste de entities |
| Router LLM fora do enum em produção | Parser já lança `ValueError` → fallback determinístico (inalterado) |
| Resposta "quer agendar?" reaparece | Gate + prompt do reply + `_route_state` nunca forçam scheduling sem sinal |

## Critérios de aceite (globais)

1. 348+ testes passam (conversation-router + voice-adapter + dashboard), suites separadas.
2. Sem `request_schedule` implícito: só verbo explícito ou `visit_interest` + gate.
3. Lead sem `favorite_property`/`visit_interest`/`deadline` **nunca** chega em scheduler.
4. "Gostei da Torre Nova" grava `favorite_property` e a próxima pergunta usa o nome.
5. `bash -n start.sh` OK (se tocar no script — não deve).

## Ordem de implementação

1. Etapa 1 (gate) → testes → commit  
2. Etapa 2 (router) → testes → commit  
3. Etapa 3 (memória) → testes → commit  
4. Etapa 4 (discovery + reply) → testes → commit  

Cada etapa: implementar, rodar suítes separadas, só então seguir.
