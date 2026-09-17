# Personas — Agente SDR Imobiliário B2B

## Persona 1: Lead B2B (Principal)

**Nome**: Diretor/Gerente de Facilities & Workplace  
**Role**: Cliente corporativo buscando espaço para operação  
**Contexto**: Empresa (PME a multinacional) precisa de espaço físico em São Paulo para expansão, transferência ou instalação de filial  
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
- Comprar expectativas de retorno (cap rate, vacância)
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
**Contexto**: Responsável pela performance comercial e operacional da W Levitt  
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

## Relações e Priorização

1. **Lead B2B** — Alvo principal do agente, recebe atendimento direto
2. **Gestor W Levitt** — Stakeholder que monitora e decide, usa dashboard
3. **Investidor PF** — Público secundário, fluxo similar mas com perguntas específicas de ROI

## Notas de Implementação

- O bot deve adaptar o tom e perguntas baseado na persona detectada
- Lead B2B recebe fluxo de qualificação comercial focado em espaço operacional
- Investidor PF recebe fluxo adicional de análise de ROI e ticket
- Gestor interage apenas via dashboard e alertas (não conversa direta com o bot)