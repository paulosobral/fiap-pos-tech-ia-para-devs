# Deployment Pipeline Questions

> Respostas derivadas do contexto afirmado (`team.md` § Deployment: sem esteira, deploy manual via `start.sh`/`stop.sh`; `project.md` § Forbidden: `NEVER deployar sem start.sh/stop.sh`; gate do `ci-pipeline`: dist.zip por Lambda + módulos HashiCorp) e do escopo feature/POC (custo domina, sem produção formal).

## Q1 — Estratégia de deploy
- A. **In-place via Terraform** (Recommended) — `terraform apply` idempotente; endpoint do API GW estável; dados DynamoDB preservados em update in-place
- B. Blue/green — exigiria 2×infra simultânea (custo) e automação que não existe sem CI/CD externo
- C. Canary gradual — sem plataforma de tráfego (Evidently/CodeDeploy) fora do escopo POC

[Answer]: A

## Q2 — Promoção de ambiente (dev → staging → prod)
- A. **Não aplicável** (Recommended) — ambiente único efêmero `poc`; deploy manual já é a promoção; custo AWS domina o escopo
- B. Staging + prod com `tfvars` — dobra o custo de monitoração (dynamo/SQS/ASR) e exige matriz de gates sem consumidor no hackathon

[Answer]: A

## Q3 — Aprovação de produção
- A. **Revisão do `terraform plan` pelo humano** (Recommended) — o prompt do `apply` é o gate; não há "produção" formal no POC
- B. Aprovação formal de tech lead + PO — aplicável a cadência de produção; sem pipeline/platform para materializar aqui

[Answer]: A

## Q4 — Rollback
- A. **Runbook em 4 trilhas** (Recommended) — R1 reconcile (plan/apply), R2 reverter zip de app (dist/archive retido), R3 git revert de `infra/*.tf`, R4 `stop.sh`+`start.sh` (teardown limpo) + checklist de validação
- B. Apenas teardown completo em qualquer falha — perde dados da sessão sem necessidade

[Answer]: A

## Q5 — Feature flags
- A. **Não aplicável ao POC** (Recommended) — toggles via variáveis de ambiente no Terraform (`environment` das Lambdas), versionadas em `infra/*.tf`; estratégia gradual (Evidently/AppConfig) fica para escopo enterprise
- B. CloudWatch Evidently/AppConfig — feature flags pagos + wiring por unit; sem consumidor neste escopo

[Answer]: A

## Q6 — Alvo do dashboard-ui (em aberto no gate do ci-pipeline)
- A. **ECS Fargate 1×t3.micro** (Recommended) — módulo `terraform-aws-modules/ecs/aws`; Streamlit container de longa duração acessível na nuvem; destruído pelo `stop.sh`
- B. Rodar apenas local (sem container/nuvem) — mais simples, mas o PRD pede o dashboard acessível em demo com `start.sh` na nuvem

[Answer]: A

## Consolidated Summary Confirmation

<!-- preenchido antes do approval gate -->

[Answer]: Looks correct