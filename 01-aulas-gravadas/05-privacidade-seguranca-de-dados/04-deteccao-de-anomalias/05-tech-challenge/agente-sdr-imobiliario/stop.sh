#!/usr/bin/env bash
# stop.sh — teardown do ambiente (NFR9.3). NEVER deployar sem passar pelos scripts (project.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/infra"

export AWS_PAGER=""

terraform init -input=false >/dev/null 2>&1 || true
if terraform state list >/dev/null 2>&1; then
  terraform destroy -auto-approve -input=false
else
  echo "Nada para destruir (nenhum estado terraform)."
fi

# Limpeza de log groups órfãos: o destroy do Terraform remove os recursos, mas a
# AWS auto-cria /aws/lambda/sdr-* quando uma Lambda é invocada por schedule entre
# o destroy e o próximo apply. Sem isso, o start.sh falha com ResourceAlreadyExistsException.
echo "Limpando log groups órfãos..."
REGION="$(terraform output -raw region 2>/dev/null || echo us-east-1)"
for prefix in "/aws/lambda/sdr-" "/ecs/sdr-"; do
  groups=$(aws logs describe-log-groups --region "$REGION" --log-group-name-prefix "$prefix" --query "logGroups[].logGroupName" --output text 2>/dev/null || true)
  if [ -n "$groups" ]; then
    for lg in $groups; do
      aws logs delete-log-group --region "$REGION" --log-group-name "$lg" 2>/dev/null && echo "  removido: $lg" || true
    done
  fi
done
echo "Limpeza concluida."