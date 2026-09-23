#!/usr/bin/env bash
# start.sh — pipeline local de deploy do Agente SDR (CI/CD do POC)
# Fases 1-4 = CI (ci-config.md); fase 5 = deploy (terraform apply); fase 6 = smoke.
# NEVER deployar sem passar por este script (project.md).
set -euo pipefail
export AWS_PAGER=""

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "== [1/6] Setup"
if [ ! -x .venv/bin/python ]; then
  echo "criando venv..."
  python3 -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
# Deps de teste do venv: gates rodam antes da fase 4 — conversation-router precisa de faiss/litellm p/ os testes dele
.venv/bin/pip install -q -r requirements-dev.txt -r apps/conversation-router/requirements.txt
PY=".venv/bin/python"

echo "== [2/6] compileall"
"$PY" -m compileall -q apps || { echo "FALHA: compile"; exit 1; }
echo "compileall OK"

echo "== [3/6] Gates (pytest por unit — mesmos comandos de test-results.md)"
for u in conversation-router voice-adapter crm-adapter contact-ingest anomaly-detector followup; do
  COVERAGE_FILE="/tmp/.cov-$u" "$PY" -m pytest "apps/$u/tests" \
    --cov="apps/$u" --cov-report=term --cov-fail-under=80 -q || { echo "FALHA: $u"; exit 1; }
done
COVERAGE_FILE="/tmp/.cov-u7" "$PY" -m pytest apps/dashboard-api/tests \
  --cov=apps/dashboard-api --cov-report=term --cov-fail-under=80 -q || { echo "FALHA: dashboard-api"; exit 1; }
"$PY" -m pytest apps/dashboard-ui/tests -q || { echo "FALHA: dashboard-ui"; exit 1; }
echo "Gates OK"

echo "== [4/6] Build dist/*.zip (7 Lambdas; dashboard-ui = container via ECR)"
mkdir -p dist dist/archive
# Catálogo sintético de imóveis (FR-11): gerado no build, data/ é gitignored
mkdir -p apps/conversation-router/data
"$PY" scripts/seed_properties.py > apps/conversation-router/data/properties.json
"$PY" scripts/seed_clients.py > apps/conversation-router/data/clients.json
echo "RAG seed: $(python3 -c 'import json;print(len(json.load(open("apps/conversation-router/data/properties.json"))["properties"]))') imóveis + $(python3 -c 'import json;print(len(json.load(open("apps/conversation-router/data/clients.json"))["clients"]))') clientes sintéticos"
for a in voice-adapter crm-adapter contact-ingest anomaly-detector followup dashboard-api; do
  if [ -f "dist/$a.zip" ]; then
    cp "dist/$a.zip" "dist/archive/$a-$(date +%Y%m%d%H%M%S).zip"
    rm -f "dist/$a.zip"
  fi
  if [ "$a" = "voice-adapter" ]; then
    # POC: transcrição embarcada (faster-whisper+ffmpeg+ctranslate2 ~>120MB) NÃO cabe no
    # pacote direto da Lambda (limite 50MB). deploy com boto3(nativo)+requests; a transcrição
    # real fica para amazon-transcribe/layer em evolução pós-POC (ver health-check-report).
    .venv/bin/pip install -q requests -t "dist/$a/" \
      --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:
  else
    .venv/bin/pip install -q -r "apps/$a/requirements.txt" -t "dist/$a/" \
      --platform manylinux2014_x86_64 --python-version 3.11 --only-binary=:all:
  fi
  find "dist/$a" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find "dist/$a" -maxdepth 2 -name '*.dist-info' -type d -exec rm -rf {} + 2>/dev/null || true
  ( cd "apps/$a" && zip -qr "../../dist/$a.zip" . )
  ( cd "dist/$a" && zip -qr "../$a.zip" . )
  rm -rf "dist/$a"
done
echo "zips OK: $(ls -1 dist/*.zip | wc -l) artefatos"

# Opcional: token do Telegram lido de secrets.local.env (gitignored) se existir
if [ -f secrets.local.env ]; then
  set -a; source secrets.local.env; set +a
fi

echo "== [5/6] Deploy (terraform init + plan + apply)"
cd infra
if ! terraform init -upgrade -input=false >/tmp/td-init.log 2>&1; then
  echo "FALHA: terraform init"; tail -20 /tmp/td-init.log; exit 1
fi

# apply 1: cria o ECR e a infra base (as tasks ainda sem imagem)
if ! terraform apply -auto-approve -input=false -var="telegram_bot_token=${TELEGRAM_BOT_TOKEN:-}" \
     -var="llm_api_key=${LLM_API_KEY:-}" \
     >/tmp/td-apply1.log 2>&1; then
  echo "FALHA: terraform apply (base)"; tail -40 /tmp/td-apply1.log; exit 1
fi
DASH_ECR_URI="$(terraform output -raw dashboard_ecr_repo)"
VOICE_ECR_URI="$(terraform output -raw voice_ecr_repo)"
ROUTER_ECR_URI="$(terraform output -raw router_ecr_repo)"
echo "ECR dashboard: $DASH_ECR_URI"
echo "ECR voice:     $VOICE_ECR_URI"
echo "ECR router:    $ROUTER_ECR_URI"
REGION="$(terraform output -raw region)"

# build + push da imagem do conversation-router (podman, litellm + langgraph + faiss completos)
router_tag="$ROUTER_ECR_URI:poc-$(date +%Y%m%d%H%M%S)"
echo "== [5a/6] build imagem conversation-router (podman, litellm+langgraph+faiss)"
podman build -q -t "$router_tag" -f ../apps/conversation-router/Dockerfile ../apps/conversation-router/ >/tmp/td-podman-router.log 2>&1 || {
  echo "FALHA: podman build conversation-router"; tail -20 /tmp/td-podman-router.log; exit 1; }
aws ecr get-login-password --region "$REGION" \
  | podman login --username AWS --password-stdin "$(echo "$ROUTER_ECR_URI" | cut -d/ -f1)" >/dev/null 2>&1
podman push -q "$router_tag" >>/tmp/td-podman-router.log 2>&1 || { echo "FALHA: push ECR conversation-router"; tail -10 /tmp/td-podman-router.log; exit 1; }
echo "imagem publicada: $router_tag"

# build + push da imagem do dashboard-ui (podman, sem docker)
dash_tag="$DASH_ECR_URI:poc-$(date +%Y%m%d%H%M%S)"
echo "== [5b/6] build imagem dashboard-ui (podman)"
podman build -q -t "$dash_tag" -f ../apps/dashboard-ui/Dockerfile ../apps/dashboard-ui/ >/tmp/td-podman.log 2>&1 || {
  echo "FALHA: podman build dashboard-ui"; tail -20 /tmp/td-podman.log; exit 1; }
aws ecr get-login-password --region "$REGION" \
  | podman login --username AWS --password-stdin "$(echo "$DASH_ECR_URI" | cut -d/ -f1)" >/dev/null 2>&1
podman push -q "$dash_tag" >>/tmp/td-podman.log 2>&1 || { echo "FALHA: push ECR dashboard-ui"; tail -10 /tmp/td-podman.log; exit 1; }
echo "imagem publicada: $dash_tag"

# build + push da imagem do voice-adapter (faster-whisper real — ECS Fargate)
voice_tag="$VOICE_ECR_URI:poc-$(date +%Y%m%d%H%M%S)"
echo "== [5c/6] build imagem voice-adapter (podman, faster-whisper + ffmpeg)"
podman build -q -t "$voice_tag" -f ../apps/voice-adapter/Dockerfile ../apps/voice-adapter/ >/tmp/td-podman-voice.log 2>&1 || {
  echo "FALHA: podman build voice-adapter"; tail -20 /tmp/td-podman-voice.log; exit 1; }
aws ecr get-login-password --region "$REGION" \
  | podman login --username AWS --password-stdin "$(echo "$VOICE_ECR_URI" | cut -d/ -f1)" >/dev/null 2>&1
podman push -q "$voice_tag" >>/tmp/td-podman-voice.log 2>&1 || { echo "FALHA: push ECR voice-adapter"; tail -10 /tmp/td-podman-voice.log; exit 1; }
echo "imagem publicada: $voice_tag"

# apply 2: aponta as task definitions para as imagens (scale-out manual p/ smoke)
if ! terraform apply -auto-approve -input=false \
     -var="telegram_bot_token=${TELEGRAM_BOT_TOKEN:-}" \
     -var="llm_api_key=${LLM_API_KEY:-}" \
     -var="dashboard_ui_image=$dash_tag" \
     -var="voice_adapter_image=$voice_tag" \
     -var="router_image=$router_tag" \
     >/tmp/td-apply2.log 2>&1; then
  echo "FALHA: terraform apply (imagens)"; tail -40 /tmp/td-apply2.log; exit 1
fi

API_URL="$(terraform output -raw api_url)"
cd ..

# Scale-out manual das 3 tasks ECS (o cron so dispara no proximo 09:00 BRT)
echo "== [5d/6] Scale-out das tasks ECS"
CLUSTER="sdr-cluster"
for svc in sdr-conversation-router sdr-dashboard-ui sdr-voice-adapter; do
  echo -n "  $svc -> 1... "
  aws ecs update-service --cluster "$CLUSTER" --service "$svc" --desired-count 1 \
    --region "$REGION" --query "service.serviceName" --output text 2>/dev/null && echo "OK" || echo "skip"
done

echo "Aguardando tasks subirem..."
for i in $(seq 1 12); do
  running=$(aws ecs describe-services --cluster "$CLUSTER" \
    --services sdr-conversation-router sdr-dashboard-ui sdr-voice-adapter \
    --region "$REGION" --query "sum(services[].runningCount)" --output text 2>/dev/null || echo 0)
  echo "  running: $running/3"
  [ "$running" = "3" ] && break
  sleep 10
done

# Aponta as integrações HTTP_PROXY do API Gateway para o IP público da task do conversation-router
echo "== [5e/6] Conectando API Gateway -> conversation-router"
ROUTER_TASK_ARN=$(aws ecs list-tasks --cluster "$CLUSTER" --service-name sdr-conversation-router \
  --region "$REGION" --query "taskArns[0]" --output text 2>/dev/null)
ROUTER_ENI=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$ROUTER_TASK_ARN" \
  --region "$REGION" --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text 2>/dev/null)
ROUTER_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$ROUTER_ENI" \
  --region "$REGION" --query "NetworkInterfaces[0].Association.PublicIp" --output text 2>/dev/null)
echo "  Router IP: $ROUTER_IP"

API_ID=$(aws apigatewayv2 get-apis --region "$REGION" \
  --query "Items[?Name=='sdr-http-api'].ApiId" --output text 2>/dev/null)
# Integração POST (webhook + internal)
POST_INT_ID=$(aws apigatewayv2 get-integrations --api-id "$API_ID" --region "$REGION" \
  --query "Items[?IntegrationType=='HTTP_PROXY' && IntegrationMethod=='POST'].IntegrationId" --output text 2>/dev/null)
aws apigatewayv2 update-integration --api-id "$API_ID" --integration-id "$POST_INT_ID" \
  --region "$REGION" --integration-uri "http://$ROUTER_IP:8080" >/dev/null 2>&1
# Integração GET (health)
GET_INT_ID=$(aws apigatewayv2 get-integrations --api-id "$API_ID" --region "$REGION" \
  --query "Items[?IntegrationType=='HTTP_PROXY' && IntegrationMethod=='GET'].IntegrationId" --output text 2>/dev/null)
aws apigatewayv2 update-integration --api-id "$API_ID" --integration-id "$GET_INT_ID" \
  --region "$REGION" --integration-uri "http://$ROUTER_IP:8080" >/dev/null 2>&1
echo "  API Gateway -> http://$ROUTER_IP:8080"

# Setar o webhook do Telegram para o API Gateway (HTTPS exigido pelo Telegram)
if [ -n "${TELEGRAM_BOT_TOKEN:-}" ]; then
  WEBHOOK_URL="$API_URL/webhook/telegram"
  SECRET_TOKEN=$(aws secretsmanager get-secret-value --region "$REGION" \
    --secret-id sdr/dashboard-api-token --query "SecretString" --output text 2>/dev/null || echo "")
  if [ -n "$SECRET_TOKEN" ]; then
    TG_RESULT=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=${WEBHOOK_URL}&secret_token=${SECRET_TOKEN}")
  else
    TG_RESULT=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=${WEBHOOK_URL}")
  fi
  echo "  Telegram webhook: $WEBHOOK_URL"
  echo "$TG_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Telegram:', 'OK' if d.get('ok') else 'FAIL: '+d.get('description',''))" 2>/dev/null || echo "  Telegram: verifique manualmente"
fi

echo "== [6/6] Smoke checks"
"$PY" - <<PYEOF
import json, urllib.request, time
api = "$API_URL"
ok = True
def check(name, fn):
    global ok
    try:
        fn(); print(f"  [OK] {name}")
    except Exception as e:
        ok = False; print(f"  [FAIL] {name}: {e}")
def hit_health():
    r = urllib.request.urlopen(api + "/health", timeout=15)
    assert r.status == 200, f"status {r.status}"
# check("GET /health", hit_health) — health agora no container conversation-router (porta 8080)
def hit_dashboard():
    req = urllib.request.Request(api + "/api/kpis", method="GET",
        headers={"Authorization": "Bearer poc-smoke"})
    r = urllib.request.urlopen(req, timeout=15)
    assert r.status == 200, f"status {r.status}"
check("GET /api/kpis", hit_dashboard)
if ok:
    print("SMOKE OK — API:", api)
else:
    print("SMOKE COM FALHAS")
    raise SystemExit(1)
PYEOF

echo
echo "Deploy concluído."
echo "API:                 $API_URL"
echo "Conversation Router: task do ECS sdr-conversation-router (porta 8080; escala 09:00-18:00 BRT)"
echo "Dashboard:           task do ECS sdr-dashboard-ui (IP público no Console > ECS > cluster sdr > service; escala 09:00-18:00 BRT)"
echo "Voice adapter:       task do ECS sdr-voice-adapter (faster-whisper; escala 09:00-18:00 BRT; consome sdr-voice-queue)"
echo "Token Telegram:      substitua em Secrets Manager (sdr/tg-bot-token) e re-aplique p/ ativar o bot"
echo "LLM (OpenRouter):    defina LLM_API_KEY em secrets.local.env e re-aplique — Terraform grava na secret sdr/llm-api-key"
echo "Teardown:            ./stop.sh"