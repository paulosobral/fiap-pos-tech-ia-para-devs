# Memory — Deployment Pipeline (operation)

> Working memory do estágio `deployment-pipeline`. Consolidado antes do approval gate.

## Interpretations

- "CD pipeline" neste projeto = o próprio `start.sh`/`stop.sh` (regra `team.md`/`project.md`); não há ferramenta externa para configurar — o estágio documenta a estratégia CD do pipeline local.
- "Deployment strategy" para POC = in-place via Terraform; blue/green/canary são não-aplicáveis e isso está documentado como decisão (não como omissão).
- A matriz de promoção dev→staging→prod foi respondida como "não aplicável" com justificativa de custo/escopo, atendendo ao pedido do stage file sem inventar ambientes.

## Deviations

- O stage file pede `consumes: infrastructure-specification` e `cicd-pipeline` — ambos os estágios estão `[S]kipped` nesta execução; consumiu-se `ci-config.md`/`quality-gates.md` (ci-pipeline) + NFR9.x do design, que cobrem o mesmo terreno no escopo atual (infra por serviço, módulos HashiCorp).
- `dashboard-ui` alvo de deploy estava em aberto no ci-config ("confirmar no deployment-pipeline") — resolvido na Q6 do questions file (ECS Fargate), aguardando confirmação do humano no gate.

## Tradeoffs

- Backend Terraform **local** (sem S3+lock): zero custo/setup no POC; contra: sem lock entre operadores e state no repo — mitigado por `.gitignore` de `infra/.terraform` e por ser deploy single-operator.
- Zero-downtime **não garantido**: aceito janela manual; troca por estabilidade de endpoint (API GW estável) e preservação de dados DynamoDB.
- Retenção de zips anteriores em `dist/archive/` (R2 do rollback): custo de disco mínimo vs. reversibilidade de código sem git-ops extra.

## Open questions

- Q6 (ECS vs local para dashboard-ui) — proposta A, confirmar no gate.
- Smoke check `/health`: rota ainda não existe nas Lambdas — implementar em `deployment-execution` (já anotado no cd-config como fase 6).