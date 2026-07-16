#!/usr/bin/env bash
# ==============================================================================
# start.sh — provisiona o bucket S3 (terraform apply) e sobe o Streamlit.
#
# 1. terraform apply -auto-approve no diretório infra/ (cria o bucket S3)
# 2. source venv/bin/activate
# 3. streamlit run app.py via nohup, com AUDIO_TRANSCRIBE_BUCKET apontando
#    para o bucket recém-criado; imprime PID e endereço de conexão.
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFRA_DIR="${ROOT_DIR}/infra"
LOG_FILE="${ROOT_DIR}/streamlit.log"
PID_FILE="${ROOT_DIR}/streamlit.pid"
PORT="${STREAMLIT_PORT:-8501}"

echo ">> Provisionando bucket S3 via Terraform..."
terraform -chdir="${INFRA_DIR}" init -input=false
terraform -chdir="${INFRA_DIR}" apply -auto-approve

BUCKET="$(terraform -chdir="${INFRA_DIR}" output -raw transcribe_bucket_name)"
echo ">> Bucket criado: ${BUCKET}"

echo ">> Ativando venv..."
# shellcheck disable=SC1091
source "${ROOT_DIR}/venv/bin/activate"

echo ">> Subindo Streamlit (nohup)..."
cd "${ROOT_DIR}"
AUDIO_TRANSCRIBE_BUCKET="${BUCKET}" \
  nohup streamlit run app.py \
    --server.port "${PORT}" \
    --server.headless true \
    > "${LOG_FILE}" 2>&1 &

STREAMLIT_PID=$!
echo "${STREAMLIT_PID}" > "${PID_FILE}"

echo ""
echo "=============================================="
echo " Streamlit rodando"
echo "   PID:      ${STREAMLIT_PID}  (salvo em ${PID_FILE})"
echo "   URL:      http://localhost:${PORT}"
echo "   Bucket:   ${BUCKET}"
echo "   Log:      ${LOG_FILE}"
echo "=============================================="
