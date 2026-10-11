# CI Config — Agente SDR Imobiliário (POC)

## Decisão (afirmada pelo responsável, 2026-09-21; confirmada pela prática)
- **Sem ferramenta de CI externa**: o GitHub é só repositório; não há `.github/workflows` (conferido em 2026-10-10). GitHub Actions foi proposto e recusado.
- **A esteira de integração e deploy são os scripts locais `start.sh` e `stop.sh`**, como no PRD e nas regras do time (`team.md` § Deployment; `project.md` § Forbidden: `NEVER` deployar fora deles).
- **Integração de código**: branch por feature (a partir de `feature/01-aulas-gravadas/05-privacidade-seguranca-de-dados`) → pull request → merge, com revisão humana e sem CI no PR.

## Pipeline do `start.sh` (estado verificado em 2026-10-10)
| Fase | O que faz |
|---|---|
| 1. Setup | cria o `.venv` e instala `requirements-dev.txt` + dependências do router |
| 2. `compileall` | valida a sintaxe de `apps/` |
| Catálogo | usa a saída limpa do crawler (`../crawling-imobiliarias`) ou, sem ela, o catálogo sintético; gera `clients.json` |
| 3. Gates | `pytest` por aplicação (sem a pasta `quality`), cobertura ≥ 80%, mais `tests/infra` (índices DynamoDB × Terraform) |
| 3b. Qualidade com LLM real | `pytest apps/conversation-router/tests/quality`, repetido 1x nos que falharem; pulado, com aviso, se não houver chave de LLM |
| 4. Build | um `dist/<app>.zip` por Lambda (`pip install --platform manylinux2014_x86_64 --python-version 3.11`; o `crm-adapter` mantém os `*.dist-info` por causa do `httpx2`/`mcp`) |
| 5. Deploy | `terraform apply` em dois passos (o 1º cria o ECR e a base; o 2º aponta as task definitions para as imagens `podman` do router, do dashboard e do voice-adapter), usuário de smoke no Cognito, escala dos serviços ECS, seed do catálogo no DynamoDB, webhook do Telegram |
| 6. Smoke | `/health` e `GET /api/kpis` com token Cognito |

Toda a saída vai para a tela e para `logs/start-AAAAMMDD-HHMMSS.log` (atalho `logs/start-latest.log`). O `stop.sh` salva o refresh token do HubSpot, faz o `terraform destroy` e limpa os log groups órfãos.

## Artefatos e IaC
- **Lambdas** (7 módulos `terraform-aws-modules/lambda/aws`, `python3.11`): `crm-adapter`, `contact-ingest`, `anomaly-detector`, `followup`, `dashboard-api`, `voice-adapter` (declarada, sem mapping de SQS) e `conversation-router` (declarada; o router roda no ECS).
- **ECS Fargate** (recursos nativos): `conversation-router`, `voice-adapter` e `dashboard-ui`, com liga/desliga agendado (09:00–18:00 BRT). O router tem o daemon do X-Ray como contêiner auxiliar.
- **Demais recursos são nativos, sem módulo da comunidade** (diferente do desenho de 2026-09-21): API Gateway HTTP (`aws_apigatewayv2_*`), DynamoDB, SQS com DLQ, EventBridge, Step Functions, Cognito, SES, Secrets Manager, SSM, KMS.
- **Repositório de artefatos**: nenhum. Os zips nascem do `start.sh`; as imagens vão para o ECR (`podman push`).

## O que mudou desde o desenho de 2026-09-21
- Gate com LLM real e guarda de índices DynamoDB entraram no `start.sh`.
- Dashboard e router passaram a ser imagens `podman` no ECR (antes: Lambdas e Streamlit à parte).
- Logs de execução em arquivo; X-Ray; HubSpot via MCP (segredo `sdr/hubspot-mcp` carregado no apply).
