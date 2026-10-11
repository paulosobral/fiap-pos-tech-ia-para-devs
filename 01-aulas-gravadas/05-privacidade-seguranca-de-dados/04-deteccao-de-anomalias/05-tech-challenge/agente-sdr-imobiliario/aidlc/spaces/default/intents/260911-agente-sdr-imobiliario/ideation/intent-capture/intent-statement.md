# Intent Statement — Agente SDR Imobiliário B2B (W Levitt)

## Problem Statement

A W Levitt negocia imóveis corporativos e comerciais em São Paulo (lajes corporativas, andares, conjuntos, salas e terrenos/edifícios), com clientes empresas que buscam espaço para operação. O primeiro atendimento e a triagem de leads dependem de horário comercial e de trabalho manual: a resposta inicial é lenta, leads parados não recebem follow-up, e o corretor especialista consome tempo com conversas brutas em vez de resumos qualificados [desc] [Q1-A] [Q1-B] [Q1-C]. Além disso, a identificação da intenção (compra/locação/investimento) e a escala no primeiro contato dependem de processo manual [Q1-D].

O Agente SDR B2B deve resolver esse conjunto de dores com atendimento conversacional automatizado 24×7, qualificação com pontuação, follow-up com contexto e handoff qualificado ao time de vendas [Q1] [desc].

## Target Customer

- **Cliente externo principal**: empresas B2B que buscam espaço corporativo em São Paulo — de PMEs a multinacionais (lajes corporativas, andares, conjuntos, salas, terrenos/edifícios para investimento) [Q2-A].
- **Parcela menor**: pessoa física investidora que quer investir em imóveis (ex.: salas para renda) [Q2-B].
- **Cliente interno (equipe de vendas)**: corretores especialistas e SDRs da W Levitt, que recebem leads qualificados e resumidos [Q2-C].
- **Beneficiário indireto**: gestor/proprietário da W Levitt, beneficiado por operação mais automatizada e assertiva [Q2-D] [Q8].

## Success Metrics

- Tempo de primeira resposta < 10 s [Q3].
- Taxa de leads qualificados ≥ 60% das conversas [Q3].
- Intenção de compra/locação/investimento corretamente detectada ≥ 85% [Q3].
  *(Metas de agendamentos e reativação por follow-up do PRD preliminar (§13, linhas 663-664) foram avaliadas e descartadas: não são KPIs adequados para demonstração; não constam do escopo desta etapa.)*

## Initiative Trigger

- **Demanda acadêmica / prazo**: hackathon da FIAP — Fase 5 do curso PosTech IA para Devs, com entrega em 12 de outubro [Q4].
- Oportunidade de mercado (atendimento B2B corporativo imobiliário com IA) permanece como contexto motivador, mas o gatilho acionado foi o prazo do hackathon [Q4] [assumption].

## Initial Scope Signal

- Workflow-selected scope: `feature` (workflow-selected) [scope].
- Limite do produto confirmado nesta etapa: atender cliente final B2B (empresa e investidor), equipe de vendas (corretor/SDR) e, de modo indireto, o gestor/proprietário da W Levitt [Q8]. O escopo funcional detalhado (canais, roleta, esteira Kanban, cenários C1–C7) será confirmado nas próximas fases [assumption].

## Assumptions & Open Questions

- Detecção de anomalias: **confirmada como objetivo da POC (plus/diferencial)** — não exigida pelo enunciado do TC (só "segurança" genérica); vem do PRD preliminar (§8.5, cenário C5, KPI l.665). Escopo: IF + PCA + GLR + Autoencoder (técnicas das aulas), job diário (Lambda), alerta no dashboard, restrição de agendamento para leads suspeitos. Confirmado pelo usuário em 2026-09-12.
- O canal de atendimento será via Telegram — única via, WhatsApp está fora do escopo [Q2-A] [Q8]; o canal interno do time será o dashboard [Q8].
- Integração com CRM (A6): **HubSpot real via MCP, numa demonstração única ao vivo para o vídeo de apresentação** — MCP auth app + MCP Inspector (OAuth/PKCE gerenciado pela ferramenta, redirect localhost). O `crm-adapter` de fundo (SQS → Lambda) usa **simulado (default) + HubSpot real via Private App token (REST)**. MCP-HubSpot em produção (adapter/agente embutido) fica no roadmap. Confirmado em 2026-09-12 — não é [assumption].
- Trilha de auditoria da roleta/esteira (A5): **confirmada no nível de consulta simples** — timeline do lead no dashboard (entrada, qualificação, rota da roleta com regra aplicada, mudanças de status, anomalias), lida do DynamoDB que o `lead-router` já grava (PRD l.243). Sem auditoria formal imutável. Confirmado em 2026-09-12.
- Reporte formal periódico (semanal/mensal): **fora da POC** — o dashboard em tempo real (FR-07/§10) cobre o agregado para o gestor; reporte periódico seria roadmap pós-POC. Decidido em 2026-09-12. [Q7]

## Review

Confirmação: usuário aprovou o resumo consolidado das perguntas (Summary Confirmation) antes da geração deste artefato.