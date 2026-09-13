# Market Trends — Agente SDR Imobiliário B2B (W Levitt)

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt …" (workflow-selected scope `feature`).
- [scope] Workflow-selected scope: `feature`.
- [Q4] — resposta do usuário em `market-research-questions.md`.
- [assumption] PRD preliminar: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md` (§2.4 insights da mentoria; §8.5 anomalias; §16 roadmap).
- [assumption] Pesquisa de mercado (2026-09-13): site da Lais.ai (lais.ai).

---

## 1. Tendências relevantes

O usuário indicou [Q4] que a tendência mais relevante para a POC é:

- **A. Adoção crescente de IA generativa em atendimento/vendas imobiliárias (2024–2026)**.

As demais tendências avaliadas (LGPD, migração para WhatsApp Business API, expectativa de resposta imediata) permanecem como contexto, mas a adoção de IA generativa é a tendência central que a POC aproveita.

## 2. Análise da tendência central: IA generativa em vendas imobiliárias

### Radar tecnológico

| Tecnologia | Posição | Justificativa |
|---|---|---|
| SDR/atendimento imobiliário com IA generativa | **Adopt** (para a POC) | Mercado em plena adoção; Lais.ai declara "mais de 1.000 imobiliárias" clientes; o Levitt.AI aproveita a maturidade da LLM (OpenRouter/Claude) |
| RAG sobre catálogo de imóveis e clientes | **Adopt** | Base do qualificador proprietário (score explicável) — diferencial do Levitt.AI |
| Detecção de anomalias em fluxo de vendas | **Trial/Assess** | Plus/diferencial da POC (IF + PCA + GLR + Autoencoder); não é table-stake de mercado ainda |
| WhatsApp Business API como canal | **Trial** (roadmap) | Canal real do cliente, mas pago e com aprovação de número — fica no roadmap pós-POC (PRD §16) |
| Omnichannel completo (Instagram, LinkedIn, site) | **Hold/Assess** | Roadmap pós-POC (PRD §16) |

### Ciclo de hype (Gartner, aplicado)

- **IA generativa em atendimento/vendas**: em ascensão de "Peak of Inflated Expectations" para "Trough of Disillusionment" — maturidade prática crescente, mas ainda com hype. Para a POC, o uso é pragmático: LLM commodity via OpenRouter, qualificação + RAG + score explicável.
- **SDR imobiliário com IA**: em "Peak of Inflated Expectations" — muitos players residenciais (Lais, Maya), nicho corporativo B2B ainda pouco atendido (oportunidade).

### Sinal de cliente (mentoria W Levitt)

A transcrição da mentoria (PRD §2.4) valida a tendência: o cliente Leonardo (diretor comercial da W Levitt) reporta **~2.000 contatos de carteira**, leads que chegam por **portais e WhatsApp**, e conversão de **3–4 a cada 100 leads de portal** — com **queima rápida** do lead de portal (vários corretores disputam o mesmo contato). Isso reforça a necessidade de **just-in-time (primeira resposta em segundos)**, que é exatamente o que a IA generativa viabiliza.

## 3. Tendências regulatórias e de conformidade

- **LGPD e proteção de dados**: a POC trata PII de leads (nome, e-mail, telefone). A LGPD é um requisito transversal: PII mascarada, guardrails e trilha de auditoria [Q4-B][intent-statement]. A detecção de anomalias também é um plus de segurança (PRD §8.5).
- **Sem mudança regulatória bloqueadora identificada** para a POC: o escopo é demonstração acadêmica (hackathon) com dados sintéticos (base JSON+S3 de imóveis corporativos sintéticos, PRD FR-11).

## 4. Table-stakes vs diferenciais

O usuário indicou [Q5] que o **table-stake** é:

- **A. Atendimento conversacional com resposta imediata e follow-up automático**.

Isso confirma que resposta imediata e follow-up são **mínimos esperados** (não diferenciais) — qualquer concorrente sério (Lais, Maya) já os entrega. Os **diferenciais** do Levitt.AI são:

- Especialização B2B corporativa (lajes/andares/salas) — nicho não atendido [Q3-A].
- Qualificação proprietária + RAG + score explicável [Q3-D].
- Detecção de anomalias [Q3-C].
- Fully serverless / baixo custo [Q3-B].

## 5. Implicações para a POC

1. **Resposta imediata e follow-up são pré-requisitos, não diferenciais** — devem estar sólidos, mas não são o argumento de venda.
2. **Aproveitar a maturidade da LLM** (OpenRouter/Claude via LiteLLM) como commodity, concentrando esforço no que diferencia: qualificação proprietária, RAG e score explicável.
3. **Posicionar no nicho corporativo B2B**, onde a adoção de IA generativa ainda é incipiente e Lais/Maya/Squad não atuam.
4. **Anomalias como plus de segurança/diferencial** (PRD §8.5, cenário C5, KPI l.665) — reforça a tese de "IA + segurança" da POC.
5. **Arquitetura canal-agnóstica** (PRD §1 l.41) para plugar WhatsApp/omnichannel no roadmap sem retrabalho.

## 6. Conclusão

A tendência central (adoção de IA generativa em vendas imobiliárias) é favorável e em plena maturidade. A POC deve montar-se sobre essa tendência com um **diferencial de nicho (corporativo B2B) e de qualidade (qualificação proprietária + score explicável + anomalias)**, tratando resposta imediata e follow-up como table-stakes obrigatórios [Q4][Q5].