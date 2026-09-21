#!/usr/bin/env bash
# stop.sh — teardown do ambiente (NFR9.3). NEVER deployar sem passar pelos scripts (project.md).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT/infra"

terraform init -input=false >/dev/null 2>&1 || true
if terraform state list >/dev/null 2>&1; then
  if [ "${AUTO:-0}" = "1" ]; then
    terraform destroy -auto-approve -input=false
  else
    terraform destroy -input=false
  fi
else
  echo "Nada para destruir (nenhum estado terraform)."
fi