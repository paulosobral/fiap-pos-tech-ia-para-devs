#!/usr/bin/env bash
# ==============================================================================
# stop.sh — mata o processo do Streamlit e destrói o bucket S3.
#
# 1. filtra/mata o processo do Streamlit (PID salvo ou por pgrep)
# 2. terraform destroy -auto-approve no diretório infra/
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
PID_FILE="${ROOT_DIR}/streamlit.pid"

echo ">> Encerrando Streamlit..."
KILLED=""

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  if kill -0 "${PID}" 2>/dev/null; then
    kill "${PID}" && KILLED="${PID}"
  fi
  rm -f "${PID_FILE}"
fi

# fallback: procura qualquer "streamlit run app.py" solto
for pid in $(pgrep -f "streamlit run app.py" || true); do
  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" && KILLED="${KILLED} ${pid}"
  fi
done

if [[ -n "${KILLED// /}" ]]; then
  echo ">> Processo(s) morto(s):${KILLED}"
else
  echo ">> Nenhum processo Streamlit encontrado."
fi

echo ">> Destruindo bucket S3 via Terraform..."
terraform -chdir="${INFRA_DIR}" destroy -auto-approve

echo ">> Concluído."
