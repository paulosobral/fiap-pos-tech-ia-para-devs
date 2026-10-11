# Relatório de Drift — Agente SDR Imobiliário (POC)

## Estado do Terraform
Estado local, ambiente efêmero: cada `start.sh` cria do zero e o `stop.sh` destrói. Por isso não há drift acumulado entre execuções; o risco de drift é só dentro de uma sessão, se alguém mexer no console da AWS.

## Divergências entre o design e o implementado (já registradas)
- O design de infraestrutura citava módulos da comunidade (`apigateway-v2`, `dynamodb-table`, `ecs`, `sqs`); o implementado usa só o módulo das Lambdas e recursos nativos para o resto.
- O dashboard e o router, antes Lambdas/Streamlit, são imagens `podman` no ECR rodando no ECS Fargate.
- `/health` ficou no container do router e não nas Lambdas.
- Índice `lead-index` também em `sdr-alerts` (faltava; corrigido e protegido por `tests/infra`).

## Como detectar
`terraform plan` mostra o diff antes de cada `apply`; `tests/infra` garante que índices usados no código existem no `.tf`.
