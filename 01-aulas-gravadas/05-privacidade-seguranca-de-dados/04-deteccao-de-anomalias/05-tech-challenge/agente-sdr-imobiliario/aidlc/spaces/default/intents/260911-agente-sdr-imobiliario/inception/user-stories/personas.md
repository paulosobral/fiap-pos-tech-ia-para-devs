# Personas — Agente SDR Imobiliário B2B

## Persona 1: Lead B2B (Principal)

**Nome**: Diretor/Gerente de Facilities & Workplace  
**Role**: Cliente corporativo buscando espaço para operação  
**Contexto**: Empresa (PME a multinacional) precisa de espaço físico em São Paulo para expansão, transferência ou instalação de filial. Diferenças de processo decisório importam: PMEs tendem a decidir mais rápido com menos aprovadores; multinacionais envolvem comitês, mais prazo, idioma e volume. Capturar ao mínimo: cidade/região desejada, metragem aproximada, nº de ocupantes, urgência, faixa de orçamento, papel do interlocutor e quem aprova a decisão.  
**Goals**:
- Encontrar espaço corporativo adequado (laje, andar, sala) com boa localização
- Obter informações técnicas precisas sobre imóveis (área, condomínio, entrega)
- Agendar visitas com especialistas rapidamente
- Receber propostas comerciais claras e competitivas

**Pain Points**:
- Tempo de resposta elevado em corretores tradicionais
- Falta de informações detalhadas disponíveis 24×7
- Dificuldade em comparar múltiplas opções simultaneamente
- Processo manual de agendamento e acompanhamento

**Prioridade**: Alta (persona principal)

---

## Persona 2: Investidor PF (Secundário)

**Nome**: Investidor Imobiliário  
**Role**: Pessoa física buscando salas para renda  
**Contexto**: Busca imóveis comerciais para investimento/locação, foco em retorno financeiro  
**Goals**:
- Identificar imóveis com bom potencial de renda
- Comparar expectativas de retorno (cap rate, vacância, período) — sempre como estimativas com fonte e ressalva
- Ter acesso a oportunidades de alto ticket
- Receber análise de viabilidade de investimento

**Pain Points**:
- Falta de análise estruturada de ROI
- Dificuldade em encontrar oportunidades de qualidade
- Ausência de acompanhamento proativo de novos investimentos
- Processo manual de qualificação de oportunidades

**Prioridade**: Média (parcela menor do público)

---

## Persona 3: Gestor W Levitt (Monitoramento)

**Nome**: Gestor de Operações / Proprietário  
**Role**: Monitora operação do agente SDR e toma decisões estratégicas  
**Contexto**: Responsável pela performance comercial e operacional da W Levitt. Usa o dashboard majoritariamente em desktop, com sessões de revisão diárias; papéis de acesso devem ser definidos (quem vê quais métricas e alertas) e cada alerta deve apontar a ação esperada  
**Goals**:
- Monitorar volume e qualidade de leads em tempo real
- Acompanhar KPIs de atendimento (tempo de resposta, qualificação)
- Identificar anomalias e comportamentos suspeitos
- Decidir sobre investimentos em marketing e operação

**Pain Points**:
- Falta de visibilidade em tempo real do funil de vendas
- Dificuldade em identificar leads de alto valor rapidamente
- Ausência de alertas automáticos para anomalias
- Processo manual de extração de métricas de múltiplas fontes

**Prioridade**: Alta (stakeholder crítico para decisões)

---

## Persona 4: Corretor Especialista (Interna)

**Nome**: Corretor Especialista / Corretor de plantão  
**Role**: Usuário interno que recebe o handoff do lead qualificado e conduz a visita/negociação  
**Contexto**: O handoff é etapa crítica do fluxo; o corretor precisa do resumo inteligente (contexto, score, intenção, urgência) para iniciar contato sem reler a conversa. Embora não seja persona principal da POC, é usuário interno com necessidades, canal, permissões e critérios de sucesso próprios (handoff completo, atempado e sem PII além do necessário)  
**Goals**:
- Receber leads qualificados com contexto completo
- Ver o que será compartilhado (dados do lead, prazo e canal de retorno)
- Consultar apenas os dados permitidos (fronteiras de acesso por papel)

**Pain Points**:
- Resumo incompleto ou com PII excessiva
- Falta de padronização no formato do handoff
- Dependência de retrabalho para qualificar novamente o lead

**Prioridade**: Alta (interna — gate de handoff)

---

## Relações e Priorização

1. **Lead B2B** — Alvo principal do agente, recebe atendimento direto
2. **Gestor W Levitt** — Stakeholder que monitora e decide, usa dashboard
3. **Investidor PF** — Público secundário, fluxo similar mas com perguntas específicas de ROI
4. **Corretor Especialista** — Usuário interno (destinatário do handoff); não conversa com o bot, mas define critérios de sucesso do handoff

## Notas de Implementação

- O bot deve adaptar o tom e perguntas baseado na persona detectada
- Lead B2B recebe fluxo de qualificação comercial focado em espaço operacional
- Investidor PF recebe fluxo adicional de análise de ROI e ticket
- Gestor interage apenas via dashboard e alertas (não conversa direta com o bot)
- Corretor Especialista é persona interna: mesmo não sendo persona principal, precisa de necessidades, canal, permissões e critérios de sucesso definidos (handoff completo e atempado, sem PII desnecessária)
- Personas internas adicionais (ex.: SDR humano, liderança) ficam fora do escopo da POC (decisão Q7)