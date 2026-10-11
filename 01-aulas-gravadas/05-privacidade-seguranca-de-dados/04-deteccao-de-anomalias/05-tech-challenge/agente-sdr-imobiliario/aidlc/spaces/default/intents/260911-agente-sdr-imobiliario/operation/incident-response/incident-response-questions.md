# Incident Response Questions

> Respostas derivadas do projeto real (falhas já vividas: 429 do DeepSeek, token do HubSpot, IP efêmero do ECS, índice do `sdr-alerts`) e do porte do POC.

## Q1 — Modos de falha mais prováveis
- A. **LLM com 429, token do HubSpot, router/dashboard fora da janela ou com IP novo, DLQ cheia, índice DynamoDB ausente** (Recommended) — cobertos nos runbooks RB1–RB6
- B. Lista genérica de falhas de nuvem

[Answer]: A

## Q2 — Escalonamento e plantão
- A. **Sem plantão**: um responsável técnico, orientação acadêmica como segundo nível (Recommended)
- B. Rodízio de plantão com Incident Manager — sem equipe para isso

[Answer]: A

## Q3 — Remediação automática
- A. **Nenhuma automática** no POC; o que existe: fallback de modelos de LLM, DLQ com 3 tentativas e reaponte do webhook no `start.sh` (Recommended)
- B. SSM Automation por runbook

[Answer]: A

## Q4 — Comunicação durante incidentes
- A. **Responsável avisa o interessado direto**; P1 com dado pessoal segue a LGPD (Recommended)
- B. Página de status pública

[Answer]: A

## Q5 — RTO e RPO
- A. **RTO = duração do `start.sh`; RPO: sessões perdidas no `stop.sh`, leads preservados no HubSpot**; sem AWS Backup (Recommended)
- B. RTO de minutos com réplica e backups — custo fora do POC

[Answer]: A

---

## Consolidated Summary Confirmation

Resumo do estágio Incident Response:
1. Sete runbooks manuais (RB1–RB7) para as falhas reais do projeto, sem SSM Automation.
2. Plano com três níveis (P1 segredo/dado pessoal, P2 bot ou CRM fora, P3 lentidão), fluxo detectar-conter-corrigir-validar-registrar.
3. Matriz de escalonamento de uma pessoa, com orientação acadêmica e fornecedores.
4. RTO = duração do `start.sh`; RPO: sessões e alertas se perdem no `stop.sh`, leads ficam no HubSpot; sem backup.

- Looks correct
- Request changes

[Answer]: Looks correct
