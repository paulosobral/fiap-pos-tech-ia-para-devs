# Intent Capture Questions

## Sources

- [desc] Initial description: "Desenvolver do zero o Agente SDR Imobiliário B2B para a W Levitt seguindo o ciclo completo do AI-DLC — Initialization → Ideation → Inception → Construction → Operation. Partir da captura de intent (problema, clientes, canais, roleta, esteira Kanban, cenários C1–C7) e produzir, em ordem, os artefatos das fases: requisitos, user stories, domínio, contrato de API, mockups e o PRD final (padrão do framework em docs/guide/00-introduction.md). O diretório já possui o PRD preliminar e a transcrição da mentoria em documentos/ para uso como fonte."
- [scope] Workflow-selected scope: `feature`.

> Regras de operação confirmadas pelo usuário em conversa:
> - Fonte principal de entrada: `documentos/POSTECH - Hacka PRD Agente_SDR_Imobiliario - Fase 5.md`.
> - Extrair o máximo que o PRD permitir; quando faltar informação necessária, retornar ao usuário para conferência antes de avançar de fase.

---

## Q1. Business problem

Qual problema de negócio o Agente SDR resolve para a W Levitt?

- A. Atendimento de primeira resposta limitado ao horário comercial em imóveis corporativos B2B (lajes, andares, salas) em SP
- B. Follow-up de leads parados depende de corretores humanos e leva semanas/meses, perdendo oportunidades
- C. Qualificação e triagem de leads consomem tempo do corretor especialista, que recebe conversa bruta em vez de resumo qualificado
- D. Não identificação de intenção (compra/locação/investimento) e falta de escala no primeiro atendimento no WhatsApp/Telegram
- E. Duas ou mais das opções acima combinam o problema central da POC
- X. Other (please specify)

[Answer]: A (como é um chatbot não se limita apenas ao horário comercial), B, C, D

## Q2. Quem é o cliente?

Qual(is) é(são) o(s) público(s)-alvo atendido(s) pelo agente (persona do cliente final)?

- A. Empresas B2B que buscam espaço corporativo em São Paulo — de PMEs a multinacionais (lajes corporativas, andares, conjuntos, salas, terrenos/edifícios)
- B. Pessoa física investidora que queira investir (salas para renda)
- C. Corretores/especialistas internos da W Levitt que recebem leads qualificados
- D. Gestor/proprietário da W Levitt (monitora operação via dashboard)
- E. Embaixo e C, com cliente externo sendo empresa (B2B) e interno a equipe de vendas
- X. Other (please specify)

[Answer]: A, B (parcela menor... Maior público corporativo), C, D

## Q3. Como medimos sucesso?

Quais métricas devem medir o sucesso da POC (aceitar/prontidão vs métricas do §13 do PRD)?

- A. Tempo de 1ª resposta < 10s; taxa de leads qualificados >= 60% das conversas; intenção detectada corretamente >= 85%
- B. As do A mais: agendamentos >= 3 por demonstração; reativação via follow-up >= 20% dos leads parados; >= 1 anomalia detectada documentada
- C. Métricas restritas a conversa (volume de leads por dia, taxa de qualificação, agendamentos) sem meta numérica
- D. Métricas de negócio definidas pelo gestor (ex.: pipeline gerado, conversão em visita, imóvel vendido)
- E. Não definido ainda — definir nesta etapa
- X. Other (especificar)

[Answer]: A

## Q4. Trigger da iniciativa

Qual(is) é(são) o(s) gatilho(s) para este trabalho agora?

- A. Oportunidade de mercado: nenhuma das referências de mercado (Lais.ai, Plaza/Maya, Squad) é especializada em imobiliário corporativo B2B
- B. Eficiência interna: custo de tempo de SDR/corretor no WhatsApp/telegram e follow-up manual
- C. Demanda acadêmica/prazo: hackathon da FIAP — Fase 5 do curso PosTech (prazo 12 de outubro)
- D. Diferencial reputacional: W Levitt se posiciona com atendimento 24×7 com IA + humano no mesmo número
- X. Other (especificar)

[Answer]: C

## Q5. Quem são os stakeholders?

Quais stakeholders o agente atende, direta ou indiretamente? (selecione os que se aplicarem; "--select all apply--")

- A. Lead — empresa/cliente corporativo B2B (comiprador de espaço p/ operação: CFO, Facilities, PME; expansão/transferência/instalação de filial)
- B. Corretor Especialista (humano) — responsável pelo fechamento; recebe lead qualificado e resumido, não conversa bruta
- C. SDR humano — hoje faz triagem; quer escala sem perder contexto
- D. Gestor/Proprietário (W Levitt) — monitora operação, decide investimento em marketing; dashboards de volume, prontidão, anomalias, custo
- E. Times de operações/plataforma — necessidade de esteira Kanban e auditoria da roleta
- X. Other (especificar)

[Answer]: D

## Q6. Decisores vs influenciadores

- A. Decisões de direção do produto (escopo, orçamento, metodologia) são do time de projeto/equipe do hackathon; stakeholder de negócio para validação é o gestor da W Levitt
- B. Decisões de operação (qualificação, roleta, handoff) são do corretor/gerente de vendas interno
- C. Decisor externo é o cliente (empresa) que decide se avançou para reunião com o especialista
- D. Não identificado ainda
- E. Não aplicável — sem mando descontra
- X. Other (especificar)

[Answer]: A

## 7. Comunicação e cadência de reporte

Um relatório/ritual para os stakeholders da W Levitt (dashboard, Kanban, alertas)?

- A. Dashboard com clientes (1 página, Streamlit) para o gestor, atualizado em tempo quase real via API de KPIs
- B. Esteira Kanban de leads com estados visíveis para equipe de vendas (como no CRM)
- C. Alertas automáticos de anomalias (urgência artificial, bot, off-platform) via dashboard
- D. Reporte periódico definido pelo gestor (ex.: semanal) — ainda a definir
- E. Nenhum — o agente reporta para humanos e serviços automatizados apenas quando necessário
- X. Other (especificar)

[Answer]: A

## Assumptions & Open Questions

None.

## Review

Este arquivo é o registro de perguntas da etapa Intent Capture. As respostas assinaladas guiarão a geração do intent-statement e da stakeholder-map.