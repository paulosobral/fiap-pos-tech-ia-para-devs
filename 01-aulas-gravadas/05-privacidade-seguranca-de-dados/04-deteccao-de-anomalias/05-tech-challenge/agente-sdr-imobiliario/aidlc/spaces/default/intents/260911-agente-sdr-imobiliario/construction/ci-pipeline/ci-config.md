# CI Config — Agente SDR Imobiliário (POC)

## Decisão (afirmada pelo humano neste estágio)
- **Sem ferramenta de CI externa** — GitHub é apenas repositório (sem Actions).
- **Esteira de integração e deploy = scripts locais `start.sh` / `stop.sh`** conforme PRD e regras do time (`team.md` § Deployment; `project.md` § Forbidden: `NEVER deployar sem passar pelos scripts start.sh/stop.sh`).
- Integração de código: branch por feature a partir de `feature/...` → **PR → merge** (sem CI no PR; revisão humana).

## Pipeline local (equivalente CI do projeto)
`start.sh` executa, nesta ordem:

```bash
# 1. Setup
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"   # deps (cache: venv reuso)

# 2. Build (equivalente compile)
.venv/bin/python -m compileall -q apps

# 3. Testes (comandos exatos registrados pelo Build and Test — test-results.md)
for u in conversation-router voice-adapter crm-adapter contact-ingest anomaly-detector followup; do
  COVERAGE_FILE=/tmp/.cov-$u .venv/bin/python -m pytest apps/$u/tests \
    --cov=apps/$u --cov-report=term --cov-fail-under=80 -q || exit 1
done
COVERAGE_FILE=/tmp/.cov-u7 .venv/bin/python -m pytest apps/dashboard-api/tests \
  --cov=apps/dashboard-api --cov-report=term --cov-fail-under=80 -q || exit 1
.venv/bin/python -m pytest apps/dashboard-ui/tests -q || exit 1        # smoke UI

# 4. Build do artefato Lambda (dist.zip por app — design confirmado pelo humano)
mkdir -p dist
for a in conversation-router voice-adapter crm-adapter contact-ingest anomaly-detector followup dashboard-api dashboard-ui; do
  .venv/bin/pip install -r "apps/$a/requirements.txt" -t "dist/$a/" --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all: -q
  ( cd apps/$a && zip -qr "../../dist/$a.zip" . ) && ( cd "dist/$a" && zip -qr "../$a.zip" . )
  rm -rf "dist/$a"
done
# → dist/<app>.zip (handler + dependências pré-compiladas), pronto para o Terraform

# 5. Deploy (após gates) — implementação completa no estágio Deployment Execution
# terraform apply (NFR9.2 — IaC por serviço; DLQ NFR4.1, autoscaling NFR8.1)
```

`stop.sh`: `terraform destroy` (NFR9.3) — teardown do ambiente.

## Artefato Lambda e módulos Terraform (decisão do humano neste gate)
- **Build**: um `dist/<app>.zip` por aplicação (8 apps; dashboard-ui = Streamlit — ver nota) com `handler.py` + dependências pré-compiladas para o runtime **python3.11** (`pip install -t` com `--platform manylinux2014_x86_64`).
- **IaC**: `infra/` com **um `.tf` por serviço** (NFR9.1) usando **módulos HashiCorp community** para reduzir código próprio:
  - [`terraform-aws-modules/lambda/aws`](https://registry.terraform.io/modules/terraform-aws-modules/lambda/aws) — as 8 Lambdas, `handler: handler.handler`, `source_path: ../../dist/<app>.zip`
  - [`terraform-aws-modules/apigateway-v2/aws`](https://registry.terraform.io/modules/terraform-aws-modules/apigateway-v2/aws) — HTTP API + rotas → Lambdas (FR1.x webhook Telegram, FR7.x dashboard API)
  - [`terraform-aws-modules/dynamodb-table/aws`](https://registry.terraform.io/modules/terraform-aws-modules/dynamodb-table/aws) — tables de sessão/leads/alertas (TTL NFR3.3, GSI lead-index)
  - [`terraform-aws-modules/sqs/aws`](https://registry.terraform.io/modules/terraform-aws-modules/sqs/aws) — filas async + **DLQ** (NFR4.1)
  - `aws_eventbridge_*` nativo — scheduler/anomalias (NFR8.1 auto scaling embutido: Lambda + API GW escalam por natureza)
- **Dashboard-ui (Streamlit)**: container de longa duração (não Lambda) — opção: ECS Fargate 1×t3.micro no POC (módulo [`terraform-aws-modules/ecs/aws`](https://registry.terraform.io/modules/terraform-aws-modules/ecs/aws)) ou rodar local; confirmar no deployment-pipeline.
- **Onde a implementação acontece**: o *design* é este documento; **`start.sh`/`stop.sh` + `infra/*.tf` + dist são implementados e executados no estágio `deployment-execution`** (após deployment-pipeline + environment-provisioning), que é o estágio operacional que materializa IaC e roda o deploy. Deferreds NFR9.1–9.4 atualizados com owner `deployment-execution` (implementação) sob design do `deployment-pipeline`.

## Triggers
| Gatilho | Ação |
|---|---|
| Manual (humano, local) | `start.sh` (build+testes+deploy) / `stop.sh` (destroy) |
| PR merge | revisão humana; suítes rodam localmente antes do merge (padrão atual do time) |
| Push em `main` | nenhum automático (sem CI externo) |

## Branch strategy (afirmada em team.md)
- Branch de feature a partir de `feature/01-aulas-gravadas/05-privacidade-seguranca-de-dados` (base do REL atual).
- Merge via PR. Sem squash direto em `main` (difere do default org.md — especialização do team prevalece).

## Artifact repositories
- **Nenhum** (afirmado): POC Python/Node; o bundle nasce do `start.sh` no destino. `ECR/CodeArtifact/S3` não aplicáveis.

<!-- Re-saved após Consolidated Summary Confirmation (2026-09-21, authorization d00a00b3) -->

