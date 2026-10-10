#!/usr/bin/env bash
# stop.sh — teardown do ambiente (NFR9.3). NEVER deployar sem passar pelos scripts (project.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log em arquivo (logs/stop-AAAAMMDD-HHMMSS.log + logs/stop-latest.log) sem parar de mostrar na
# tela. Cores ANSI saem só do arquivo. Nenhum segredo é impresso por este script.
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/stop-$(date +%Y%m%d-%H%M%S).log"
ln -sf "$(basename "$LOG_FILE")" "$LOG_DIR/stop-latest.log"
exec > >(tee >(sed -u 's/\x1b\[[0-9;?]*[A-Za-z]//g' >> "$LOG_FILE")) 2>&1
trap 'sleep 0.5' EXIT  # deixa o tee terminar de gravar antes do prompt voltar
echo "log: $LOG_FILE"
cd "$ROOT/infra"

export AWS_PAGER=""

# O refresh token do HubSpot é de uso único e o crm-adapter o renova na secret sdr/hubspot-mcp:
# o valor do secrets.local.env fica velho. Salva o vigente ANTES do destroy para o próximo start.sh.
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
HS_JSON=$(aws secretsmanager get-secret-value --region "$REGION" --secret-id sdr/hubspot-mcp \
  --query SecretString --output text 2>/dev/null || true)
if [ -n "$HS_JSON" ] && [ -f "$ROOT/secrets.local.env" ]; then
  HS_JSON="$HS_JSON" ENV_FILE="$ROOT/secrets.local.env" python3 - <<'PY' && echo "HubSpot: refresh token vigente salvo em secrets.local.env" || echo "HubSpot: não consegui salvar o refresh token (rode scripts/hubspot_authorize.py antes do próximo start.sh)"
import json, os, re
token = json.loads(os.environ["HS_JSON"]).get("refresh_token")
assert token
path = os.environ["ENV_FILE"]
text = open(path, encoding="utf-8").read()
text = re.sub(r"^HUBSPOT_MCP_REFRESH_TOKEN=.*$", lambda _m: "HUBSPOT_MCP_REFRESH_TOKEN=" + token, text, flags=re.M)
open(path, "w", encoding="utf-8").write(text)
PY
fi

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
for prefix in "/aws/lambda/sdr-" "/ecs/sdr-"; do
  groups=$(aws logs describe-log-groups --region "$REGION" --log-group-name-prefix "$prefix" --query "logGroups[].logGroupName" --output text 2>/dev/null || true)
  if [ -n "$groups" ]; then
    for lg in $groups; do
      aws logs delete-log-group --region "$REGION" --log-group-name "$lg" 2>/dev/null && echo "  removido: $lg" || true
    done
  fi
done
echo "Limpeza concluida."