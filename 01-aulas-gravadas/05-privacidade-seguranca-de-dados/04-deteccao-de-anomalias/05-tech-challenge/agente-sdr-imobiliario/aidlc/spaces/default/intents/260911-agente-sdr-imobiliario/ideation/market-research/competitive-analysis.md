# Competitive Analysis — Agente SDR Imobiliário B2B (W Levitt)

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt …" (workflow-selected scope `feature`).
- [scope] Workflow-selected scope: `feature`.
- [Q1] [Q2] [Q3] [Q5] [Q8] — respostas do usuário em `market-research-questions.md`.
- [assumption] PRD preliminar: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md` (§1 diferenciais competitivos; §3 fora de escopo; §16 roadmap).
- [assumption] Pesquisa de mercado (2026-09-13): site da Lais.ai (lais.ai).

---

## 1. Concorrentes diretos

Concorrentes diretos são produtos que entregam a mesma categoria de solução (SDR/atendimento imobiliário com IA) ao mesmo perfil de cliente (imobiliárias). Segundo o usuário [Q1], os concorrentes diretos mais relevantes são:

| Concorrente | Foco | Canal | Segmento | Posição |
|---|---|---|---|---|
| **Lais.ai** | SDR/atendimento imobiliário com IA | WhatsApp | Imobiliárias (residencial predominante) | Líder de mercado declarado ("mais de 1000 imobiliárias") |
| **Maya (PLAZA)** | Atendimento omnichannel com IA | WhatsApp omnichannel | Imobiliárias (residencial) | Concorrente direto no atendimento |
| **Squad** | Atendimento/vendas com IA | Multi-segmento | Genérico (diversos segmentos) | Concorrente de conhecimento — mais genérico, não especializado em imobiliário |

O usuário recomendou focar a análise em **Lais.ai e Maya** (ambos voltados ao público imobiliário) e citou **Squad** apenas como referência de conhecimento, por atender diversos segmentos de forma mais genérica [Q1].

### Lais.ai (detalhe)

A Lais.ai posiciona-se como "a plataforma de IA que acompanha sua imobiliária do primeiro contato ao negócio fechado", com mais de 1.000 imobiliárias clientes. Fluxo declarado:

1. **Pré-atendimento e qualificação** — cada lead atendido em segundos, 24h/dia, com recomendação personalizada de imóveis, qualificação em escala, reengajamentos automáticos e envio direto ao CRM.
2. **Visitas** — registra cada pedido de visita no WhatsApp e centraliza a gestão de visitas programadas.
3. **Atendimento administrativo** — trata fluxos administrativos/operacionais além da qualificação de leads.

Pontos fortes da Lais: maturidade de produto, base instalada relevante, foco em conversão de leads residenciais via WhatsApp, integração com CRM e gestão de visitas. Pontos fracos (para o nosso caso): foco predominante em **imobiliárias residenciais**, sem especialização declarada em **imóveis corporativos/comerciais B2B** (lajes, andares, conjuntos, salas).

## 2. Concorrentes indiretos / substitutos

O usuário indicou [Q2] que o substituto mais relevante é **C — planilhas + trabalho manual de SDR/corretor (status quo)**. Os demais substitutos considerados:

| Substituto | Descrição | Relevância p/ W Levitt |
|---|---|---|
| **Planilhas + trabalho manual** | SDR/corretor responde e qualifica manualmente | **Alto** — é o status quo atual da W Levitt (primeira resposta limitada ao horário comercial, follow-up manual) [Q2-C] |
| CRMs com automação de follow-up | HubSpot, Pipedrive, RD Station | Médio — a W Levitt usa CRM, mas sem atendimento conversacional automatizado |
| Portais imobiliários com chat próprio | Zap, VivaReal, OLX capturam o lead antes da imobiliária | Médio — competem pela atenção do lead no primeiro contato |
| Telemarketing/tribo terceirizada | Qualificação por serviço terceirizado | Baixo — custo alto, sem escala 24×7 |

O substituto dominante é o **processo manual**: a dor central da W Levitt (lentidão da primeira resposta, follow-up manual, corretor recebendo conversa bruta) é exatamente o que o Agente SDR automatiza [Q1-Q2] [intent-statement].

## 3. Matriz de comparação de capacidades

| Capacidade | Levitt.AI (POC) | Lais.ai | Maya (PLAZA) | Squad |
|---|---|---|---|---|
| Atendimento conversacional 24×7 | Forte | Forte | Forte | Forte |
| Qualificação de leads | Forte (proprietária + score explicável) | Forte | Adequado | Adequado |
| Especialização imobiliário corporativo B2B | **Forte (nicho)** | Fraco (residencial) | Fraco (residencial) | Fraco (genérico) |
| RAG sobre catálogo de imóveis | Forte | Adequado | Adequado | Adequado |
| Detecção de anomalias | **Forte (diferencial)** | Ausente | Ausente | Ausente |
| Follow-up automático com contexto | Forte | Adequado | Adequado | Adequado |
| Handoff qualificado ao corretor | Forte | Adequado | Adequado | Adequado |
| Fully serverless / baixo custo | **Forte (~R$ 15/mês)** | Não declarado | Não declarado | Não declarado |
| Custo de operação | Baixo (serverless, OpenRouter) | Licença de produto | Licença (ou módulos) | Licença |

## 4. Mapa de posicionamento

Dois eixos que importam para o cliente-alvo:

- **Eixo X — Especialização**: genérico ↔ imobiliário corporativo B2B
- **Eixo Y — Custo/controle**: licença de produto ↔ fully serverless / baixo custo

Quadrante **pouco atendido (oportunidade)**: **especialização em imobiliário corporativo B2B + baixo custo/controle serverless**. Lais e Maya ocupam o quadrante "imobiliário residencial + licença de produto"; Squad ocupa "genérico + licença". O Levitt.AI POC posiciona-se no quadrante vazio: **especialista B2B corporativo, serverless e de baixo custo** [Q3].

## 5. SWOT — Levitt.AI

| | Helpful | Harmful |
|---|---|---|
| **Internal** | **Forças**: especialização B2B corporativa (lajes/andares/salas); qualificação proprietária + RAG + score explicável; detecção de anomalias; fully serverless de baixo custo | **Fraquezas**: POC (escopo limitado); sem base instalada; canal inicial Telegram (substituto de custo da POC) |
| **External** | **Oportunidades**: nicho B2B corporativo não atendido por Lais/Maya/Squad; adoção crescente de IA generativa em vendas imobiliárias; roadmap WhatsApp/omnichannel/multi-tenant | **Ameaças**: Lais/Maya podem expandir para corporativo; portais capturam lead antes; custo de licença de concorrentes pode baixar |

### SWOT acionável

- **Forças + Oportunidades** (perseguir): especialização corporativa B2B + adoção de IA → posicionar como o SDR de IA para imobiliário comercial, nicho vazio.
- **Fraquezas + Ameaças** (mitigar): POC sem base instalada + concorrentes expandindo → entregar POC sólida e demonstrável no hackathon; manter arquitetura canal-agnóstica para plugar WhatsApp no roadmap (PRD §1 l.41, §16).
- **Forças + Ameaças** (defensivo): baixo custo serverless + risco de licença de concorrentes → destacar custo de operação como vantagem competitiva.
- **Fraquezas + Oportunidades** (investir): POC limitada + nicho corporativo → focar a demonstração no caso B2B corporativo, que os concorrentes não cobrem.

## 6. Diferenciação (posicionamento da POC)

O posicionamento competitivo central da POC, segundo o usuário [Q3], é:

- **B. Fully serverless e baixo custo (~R$ 15/mês em POC via OpenRouter)** — vantagem de custo/controle.
- **D. Qualificação proprietária + RAG + score explicável** — diferencial de qualidade (não é só fluxo fixo).

Complementados pelo contexto [Q3][Q5]: atendimento 24×7 com IA + humano no mesmo número, e a especialização **B2B corporativa** como nicho não atendido por Lais/Maya/Squad.

**Declaração de posicionamento (POC):** "O Levitt.AI é o SDR de IA especializado em imóveis corporativos/comerciais B2B — qualificação proprietária com score explicável, detecção de anomalias e custo de operação fully serverless — para a W Levitt responder em segundos, 24×7, e entregar leads qualificados ao corretor."

## 7. Conclusão

- **Oportunidade real de nicho**: nenhum concorrente direto (Lais, Maya, Squad) é especializado em imobiliário corporativo/comercial B2B [Q1][Q3].
- **Substituto dominante**: processo manual (planilhas + SDR/corretor) — a dor que o Agente SDR ataca [Q2].
- **Diferenciais sustentáveis da POC**: especialização B2B corporativa, qualificação proprietária + score explicável, detecção de anomalias e baixo custo serverless [Q3][Q5].
- **Objetivo de mercado da POC**: posicionar como diferencial competitivo do hackathon (demonstração/avaliação) [Q8-B]; go-to-market (WhatsApp, omnichannel, multi-tenant) fica no roadmap pós-POC (PRD §16).