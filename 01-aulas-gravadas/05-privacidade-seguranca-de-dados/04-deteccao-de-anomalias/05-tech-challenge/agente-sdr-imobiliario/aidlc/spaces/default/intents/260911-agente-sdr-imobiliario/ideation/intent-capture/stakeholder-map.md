# Stakeholder Map — Agente SDR Imobiliário B2B (W Levitt)

## Key Stakeholders

| Instrumento | Interesses | Classificação | Canal preferido | Source |
|---|---|---|---|---|
| Lead B2B — empresa (CFO, Facilities/Workplace, Dir. de Expansão, Dono de PME) | Atendimento rápido, respostas técnicas precisas sobre imóveis corporativos, agendamento sem atrito | Cliente externo (beneficiário direto) | Telegram (único canal) | [Q2-A] [Q5 via Q8] |
| Investidor PF (parcela menor) | Investir em imóveis (ex.: salas para renda) com ficha qualificada | Cliente externo (beneficiário direto) | Bot de atendimento | [Q2-B] |
| Corretor Especialista (humano) | Receber lead qualificado e resumido, não conversa bruta; fechamento | Decisor de operação (equipe de vendas) | Dashboard (canal interno) | [Q2-C] [Q8] |
| SDR humano | Escala de atendimento sem perder contexto (pré-triagem) | Influe operação (equipe de vendas) | Dashboard (canal interno) | [Q8] |
| Gestor / Proprietário W Levitt | Operação mais automatizada e assertiva; dashboard de volume, prontidão, anomalias, custo | Decisor de negócio (indireto) | Dashboard (1 página) + API de KPIs | [Q2-D] [Q7] [Q8] |

*(Canal externo = Telegram (único); canal interno do time = dashboard. Integração com CRM: **HubSpot real via MCP numa demonstração única ao vivo** (vídeo de apresentação, MCP auth app + MCP Inspector) e `crm-adapter` de fundo `simulado` + HubSpot privado App (REST token) — confirmado em 2026-09-12.)*

## Decision Makers vs Influencers

- **Decisor de produto/escopo e metas**: equipe do hackathon/FIAP, com validação do gestor da W Levitt [Q6-A].
- **Decisor de operação**: equipe de vendas (corretor/gerente comercial) para qualificação, roleta e handoff [Q6-B].
- **Decisor externo**: o cliente (empresa) que decide avançar para reunião com o especialista [desc] [assumption].
- **Influenciadores**: SDR humano (pré-triagem e escala), investidor (parcela menor), equipe de plataforma/infra [Q2-B] [Q8].

## Communication Requirements

| | Descrição | Source |
|---|---|---|
| Dashboard ao gestor | Painel de 1 página (Streamlit) atualizado em tempo quase real via API de KPIs | [Q7-A] |
| Handoff ao corretor | Resumo qualificado do lead (mensagem + arquivo) no canal do time | [Q1-C] [assumption] |
| Frequência de reportes formais | **Fora da POC** — o dashboard em tempo real cobre o agregado; reporte formal periódico (semanal/mensal) fica no roadmap pós-POC | [Q7] [assumption] |

## Assumptions & Open Questions

- Canal externo = Telegram (único) [Q2-A] [Q8]; canal interno do time = dashboard [Q8]. Integração CRM confirmada em 2026-09-12 (A6): HubSpot. MCP numa demo única ao vivo (vídeo) + adapter simulado/Privado App (REST) — não é [assumption].
- Cadência de reporte formal (semanal, mensal): **fora da POC** — o dashboard cobre o agregado; roadmap pós-POC [Q7] [assumption].
- Trilha de auditoria da roleta/esteira: **confirmada no nível de consulta simples** — timeline do lead no dashboard (entrada, qualificação, rota, status, anomalias), lida do DynamoDB gravado pelo `lead-router` (PRD l.243); sem auditoria formal imutável. Confirmado em 2026-09-12 — não é [assumption].

## Review

Confirmação: usuário aprovou resumo consolidado e escopo do stakeholder map (Q8: cobertura abrangente — cliente final, equipe de vendas, gestor indireto).