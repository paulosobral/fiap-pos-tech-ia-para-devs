# CD Config — Agente SDR Imobiliário (POC)

> Consumes: `construction/ci-pipeline/ci-config.md` (pipeline local `start.sh`/`stop.sh`), `construction/ci-pipeline/quality-gates.md`, NFR9.1–9.4 (IaC), NFR4.1 (DLQ), NFR8.1 (autoscaling). Regra `project.md`: `NEVER deployar sem passar pelos scripts start.sh/stop.sh`.

## Pipeline CD (local, única via de deploy)

O deploy é manual, acionado pelo humano, e reutiliza a mesma esteira do CI local — não há ferramenta externa (GitHub é só repositório; sem Actions/CodePipeline). O `start.sh` é o *pipeline de deploy*: build → testes (gates) → `terraform apply`.

```bash
# start.sh — fluxo completo (fases 1–4 no ci-config.md; fase 5 abaixo)
# 5. Deploy
cd infra/
terraform init          # backend local (POC); sem S3 backend
terraform plan          # revisão humana é a "aprovação de produção" do POC
terraform apply         # materializa: Lambdas (dist/*.zip), HTTP API, DynamoDB, SQS+DLQ, ECS (dashboard-ui)
# 6. Smoke check pós-deploy
curl -s "$API_URL/health" || echo "FAIL: API não respondeu"    # rota de verificação (implementa em deployment-execution)
.venv/bin/python -m pytest apps/conversation-router/tests -q   # smoke mínimo reutilizando suíte existente
```

`stop.sh` = `terraform destroy` (NFR9.3) — desmonta o ambiente efêmero inteiro.

## Ambientes e promoção

| Ambiente | Uso | Promoção |
|---|---|---|
| `poc` (único, efêmero) | hackathon/demo — criado por `start.sh`, destruído por `stop.sh` | n/a — não há cadeia dev→staging→prod neste escopo |

- Um único ambiente: o deploy manual **já é a promoção** (não existe `staging` separado; custo AWS é a restrição dominante do POC).
- Matriz de promoção dev→staging→prod: **não aplicável** — documentada como não-applicável (escopo feature/POC; se o projeto evoluir, redeploy por `tfvars` por ambiente é o caminho natural).

## Aprovação

- O passo `terraform plan` + confirmação do humano no `start.sh` (prompt `Do you want to perform these actions?`) é o gate de aprovação. Nenhuma aprovação extra de tech lead/PO é exigida — não há produção formal neste POC.

## Feature flags

- **Não aplicável ao POC**: sem CloudWatch Evidently/AppConfig; toggles de comportamento (ex.: desligar anomalias) ficam em variáveis de ambiente do Terraform (`aws_lambda_function.environment`), versionadas em `infra/*.tf`. Estratégia de flags gradual fica para escopo enterprise.

## Owner da implementação

- Este estágio é **design**. `start.sh`/`stop.sh`, `infra/*.tf`, `dist/` e o smoke check são implementados e executados no **`deployment-execution`** (junto com environment-provisioning), conforme decisão registrada no gate do `ci-pipeline`.

<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21, authorization d8572d71) -->

## Atualização (2026-10-11): o deploy como ele é hoje
O desenho acima continua valendo (único caminho = `start.sh`/`stop.sh`, ambiente `poc` efêmero, sem promoção entre ambientes). O que mudou na prática:
- **Aprovação do deploy**: o `start.sh` roda `terraform apply -auto-approve`; não existe prompt de confirmação no `apply`. O gate é o humano decidir rodar o `start.sh`, depois dos gates automáticos da fase 3 (testes, cobertura ≥ 80%, `tests/infra`) e do gate de qualidade com LLM real (3b).
- **Fase 5 em dois passos**: `apply` 1 cria ECR e a base; o `podman` constrói e publica as imagens do `conversation-router`, `dashboard-ui` e `voice-adapter`; `apply` 2 aponta as task definitions para elas.
- **ECS**: Fargate (não `t3.micro`), com recursos nativos do provedor AWS (sem módulo `ecs`), liga/desliga agendado 09:00–18:00 BRT. O `dashboard-ui` e o `conversation-router` são imagens no ECR (não `dist/dashboard-ui.zip`).
- **Smoke (fase 6)**: `GET /health` (container do router) e `GET /api/kpis` com token Cognito do usuário de smoke. A suíte `pytest` não roda após o deploy; ela já passou na fase 3.
- **Log**: toda a saída vai também para `logs/start-*.log` e `logs/stop-*.log`.

