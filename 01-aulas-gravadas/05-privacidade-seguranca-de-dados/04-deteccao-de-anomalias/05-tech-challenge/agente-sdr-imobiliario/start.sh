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

# Catálogo sintético de imóveis/clientes (FR-11): precisa ANTES dos gates, porque os
# testes de conversation-router leem data/*.json (data/ é gitignored, então é gerado
# aqui). Gerar depois do pytest fazia o gate falhar em clone limpo.
mkdir -p apps/conversation-router/data
# Catálogo de imóveis: SEMPRE parte do que o crawler gerou, se existir. A cada subida o
# crawler.output passa por drop_generic_images.py (tira a colagem das fachadas da
# imobiliária e o ícone "sem foto"; cache em output/, então só baixa foto nova) e o
# resultado vira o catálogo do bot. Sem crawler -> catálogo sintético (sem fotos).
CATALOG="apps/conversation-router/data/properties.json"
CRAWLER_DIR="${CRAWLER_DIR:-$ROOT/../crawling-imobiliarias}"
CRAWLER_MERGED="$CRAWLER_DIR/output/properties_merged.json"
CRAWLER_CLEAN="$CRAWLER_DIR/output/properties_clean.json"
use_synthetic_catalog() {
  [ -f "$CATALOG" ] && cp "$CATALOG" "$CATALOG.bak"
  "$PY" scripts/seed_properties.py > "$CATALOG"
}
if [ -f "$CRAWLER_MERGED" ]; then
  echo "Catálogo do crawler encontrado ($CRAWLER_MERGED): removendo fotos genéricas..."
  if "$PY" "$CRAWLER_DIR/scripts/drop_generic_images.py" "$CRAWLER_MERGED" -o "$CRAWLER_CLEAN"; then
    cp "$CRAWLER_CLEAN" "$CATALOG"
  elif [ -f "$CRAWLER_CLEAN" ]; then
    echo "AVISO: limpeza de fotos falhou; usando a última versão limpa ($CRAWLER_CLEAN)"
    cp "$CRAWLER_CLEAN" "$CATALOG"
  else
    echo "AVISO: limpeza de fotos falhou e não há versão limpa; usando catálogo sintético (sem fotos)"
    use_synthetic_catalog
  fi
else
  echo "Sem catálogo do crawler em $CRAWLER_MERGED: usando o sintético (sem fotos)"
  use_synthetic_catalog
fi
"$PY" scripts/seed_clients.py > apps/conversation-router/data/clients.json
echo "RAG seed: $("$PY" -c 'import json;print(len(json.load(open("apps/conversation-router/data/properties.json"))["properties"]))') imóveis + $("$PY" -c 'import json;print(len(json.load(open("apps/conversation-router/data/clients.json"))["clients"]))') clientes sintéticos"

echo "== [3/6] Gates (pytest por unit — mesmos comandos de test-results.md)"
for u in conversation-router voice-adapter crm-adapter contact-ingest anomaly-detector followup; do
  COVERAGE_FILE="/tmp/.cov-$u" "$PY" -m pytest "apps/$u/tests" --ignore="apps/$u/tests/quality" \
    --cov="apps/$u" --cov-report=term --cov-fail-under=80 -q || { echo "FALHA: $u"; exit 1; }
done
COVERAGE_FILE="/tmp/.cov-u7" "$PY" -m pytest apps/dashboard-api/tests \
  --cov=apps/dashboard-api --cov-report=term --cov-fail-under=80 -q || { echo "FALHA: dashboard-api"; exit 1; }
"$PY" -m pytest apps/dashboard-ui/tests -q || { echo "FALHA: dashboard-ui"; exit 1; }
echo "Gates OK"

# Teste de qualidade com LLM real (plano "naturalidade do bot SDR", seção 5) — opcional,
# só roda quando há chave configurada (custa crédito real no OpenRouter, Tier 1); sem
# chave, não bloqueia o deploy, só avisa.
if [ -n "${OPENROUTER_API_KEY:-}${LLM_API_KEY:-}" ]; then
  echo "== [3b/6] Quality gate (LLM real, apps/conversation-router/tests/quality)"
  # LLM real não é determinística: uma falha isolada é repetida 1x (--lf roda só as
  # que falharam). Regressão de verdade falha de novo e bloqueia o deploy.
  if ! "$PY" -m pytest apps/conversation-router/tests/quality -q; then
    echo "Quality gate: repetindo 1x só os testes que falharam..."
    "$PY" -m pytest apps/conversation-router/tests/quality -q --lf || { echo "FALHA: quality gate"; exit 1; }
  fi
else
  echo "== [3b/6] Quality gate pulado (sem OPENROUTER_API_KEY/LLM_API_KEY no ambiente)"
fi

echo "== [4/6] Build dist/*.zip (7 Lambdas; dashboard-ui = container via ECR)"
mkdir -p dist dist/archive
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

# Limpeza de segurança de log groups órfãos que possam ter sobrado de execuções anteriores
echo "Verificando log groups órfãos antes do apply..."
CLEANUP_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
# Captura o state ANTES de filtrar: `terraform state list | grep -q` com pipefail
# dava falso negativo (grep -q sai cedo -> SIGPIPE no terraform -> pipeline != 0)
# e apagava os log groups ATIVOS a cada deploy.
TF_STATE_LIST="$(terraform state list 2>/dev/null || true)"
for prefix in "/aws/lambda/sdr-" "/ecs/sdr-"; do
  groups=$(aws logs describe-log-groups --region "$CLEANUP_REGION" --log-group-name-prefix "$prefix" --query "logGroups[].logGroupName" --output text 2>/dev/null || true)
  if [ -n "$groups" ]; then
    for lg in $groups; do
      # Só apaga se não estiver gerenciado no state atual
      if ! grep -q "aws_cloudwatch_log_group" <<<"$TF_STATE_LIST"; then
        aws logs delete-log-group --region "$CLEANUP_REGION" --log-group-name "$lg" 2>/dev/null && echo "  removido órfão: $lg" || true
      fi
    done
  fi
done

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

# Login no ECR com retry: falha transitória do login matava o script sem mensagem
# (linha sem tratamento + set -e) — visto no build do voice-adapter, exit 125.
ecr_login() {
  local registry
  registry="$(echo "$1" | cut -d/ -f1)"
  for attempt in 1 2 3; do
    if aws ecr get-login-password --region "$REGION" \
        | podman login --username AWS --password-stdin "$registry" >/dev/null 2>&1; then
      return 0
    fi
    echo "  login no ECR falhou (tentativa $attempt/3), tentando de novo..."
    sleep 5
  done
  echo "FALHA: podman login no ECR ($registry)"; exit 1
}

# build + push da imagem do conversation-router (podman, litellm + langgraph + faiss completos)
router_tag="$ROUTER_ECR_URI:poc-$(date +%Y%m%d%H%M%S)"
echo "== [5a/6] build imagem conversation-router (podman, litellm+langgraph+faiss)"
podman build -q -t "$router_tag" -f ../apps/conversation-router/Dockerfile ../apps/conversation-router/ >/tmp/td-podman-router.log 2>&1 || {
  echo "FALHA: podman build conversation-router"; tail -20 /tmp/td-podman-router.log; exit 1; }
ecr_login "$ROUTER_ECR_URI"
podman push -q "$router_tag" >>/tmp/td-podman-router.log 2>&1 || { echo "FALHA: push ECR conversation-router"; tail -10 /tmp/td-podman-router.log; exit 1; }
echo "imagem publicada: $router_tag"

# build + push da imagem do dashboard-ui (podman, sem docker)
dash_tag="$DASH_ECR_URI:poc-$(date +%Y%m%d%H%M%S)"
echo "== [5b/6] build imagem dashboard-ui (podman)"
podman build -q -t "$dash_tag" -f ../apps/dashboard-ui/Dockerfile ../apps/dashboard-ui/ >/tmp/td-podman.log 2>&1 || {
  echo "FALHA: podman build dashboard-ui"; tail -20 /tmp/td-podman.log; exit 1; }
ecr_login "$DASH_ECR_URI"
podman push -q "$dash_tag" >>/tmp/td-podman.log 2>&1 || { echo "FALHA: push ECR dashboard-ui"; tail -10 /tmp/td-podman.log; exit 1; }
echo "imagem publicada: $dash_tag"

# build + push da imagem do voice-adapter (faster-whisper real — ECS Fargate)
voice_tag="$VOICE_ECR_URI:poc-$(date +%Y%m%d%H%M%S)"
echo "== [5c/6] build imagem voice-adapter (podman, faster-whisper + ffmpeg)"
podman build -q -t "$voice_tag" -f ../apps/voice-adapter/Dockerfile ../apps/voice-adapter/ >/tmp/td-podman-voice.log 2>&1 || {
  echo "FALHA: podman build voice-adapter"; tail -20 /tmp/td-podman-voice.log; exit 1; }
ecr_login "$VOICE_ECR_URI"
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
POOL_ID="$(terraform output -raw cognito_user_pool_id)"
CLIENT_ID="$(terraform output -raw cognito_app_client_id)"
cd ..

# Usuário dedicado de smoke-test no Cognito (idempotente: recria a senha a cada
# run) — o smoke check de /api/kpis precisa de um JWT real agora que a rota
# está protegida pelo authorizer Cognito (ADR-014); um Bearer fixo/inventado é
# corretamente rejeitado com 401.
echo "== [5f/6] Usuário de smoke-test no Cognito"
SMOKE_USER="smoke-test@sdr.local"
SMOKE_PASSWORD="Smoke-Test-$(date +%Y)!"
aws cognito-idp admin-create-user --user-pool-id "$POOL_ID" --username "$SMOKE_USER" \
  --user-attributes Name=email,Value="$SMOKE_USER" Name=email_verified,Value=true \
  --message-action SUPPRESS --region "$REGION" >/dev/null 2>&1 || true
aws cognito-idp admin-set-user-password --user-pool-id "$POOL_ID" --username "$SMOKE_USER" \
  --password "$SMOKE_PASSWORD" --permanent --region "$REGION" >/dev/null 2>&1 || true
SMOKE_TOKEN=$(aws cognito-idp initiate-auth --auth-flow USER_PASSWORD_AUTH \
  --client-id "$CLIENT_ID" \
  --auth-parameters USERNAME="$SMOKE_USER",PASSWORD="$SMOKE_PASSWORD" \
  --region "$REGION" --query "AuthenticationResult.IdToken" --output text 2>/dev/null || echo "")
if [ -z "$SMOKE_TOKEN" ] || [ "$SMOKE_TOKEN" = "None" ]; then
  echo "AVISO: não consegui obter token Cognito pro smoke-test — /api/kpis vai dar 401 (esperado sem token válido)"
  SMOKE_TOKEN="sem-token"
fi

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

# Popula a tabela DynamoDB sdr-properties a partir de data/properties.json (idempotente).
echo "== [5d.5/6] Seed do catálogo de imóveis no DynamoDB"
PROPERTIES_TABLE="sdr-properties" AWS_REGION="$REGION" "$PY" scripts/load_properties_dynamodb.py \
  || echo "AVISO: falha ao popular sdr-properties (RAG cai para catálogo local em memória)"

# IP público da task do dashboard (Streamlit, porta 80)
DASHBOARD_TASK_ARN=$(aws ecs list-tasks --cluster "$CLUSTER" --service-name sdr-dashboard-ui \
  --region "$REGION" --query "taskArns[0]" --output text 2>/dev/null || true)
DASHBOARD_ENI=$(aws ecs describe-tasks --cluster "$CLUSTER" --tasks "$DASHBOARD_TASK_ARN" \
  --region "$REGION" --query "tasks[0].attachments[0].details[?name=='networkInterfaceId'].value" --output text 2>/dev/null || true)
DASHBOARD_IP=$(aws ec2 describe-network-interfaces --network-interface-ids "$DASHBOARD_ENI" \
  --region "$REGION" --query "NetworkInterfaces[0].Association.PublicIp" --output text 2>/dev/null || true)
DASHBOARD_URL="http://${DASHBOARD_IP:-pending}"

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
# Cada rota tem sua própria integração HTTP_PROXY com o path de destino
# embutido na URI (URI sem path sempre chama o backend em "/", quebrando o
# roteamento interno do handler). Resolve o integration-id pela rota, não
# por tipo+método, já que agora há 2 integrações POST.
update_route_integration() {
  local route_key="$1" target_path="$2" target int_id
  target=$(aws apigatewayv2 get-routes --api-id "$API_ID" --region "$REGION" \
    --query "Items[?RouteKey=='$route_key'].Target" --output text 2>/dev/null)
  int_id="${target#integrations/}"
  if [ -n "$int_id" ]; then
    aws apigatewayv2 update-integration --api-id "$API_ID" --integration-id "$int_id" \
      --region "$REGION" --integration-uri "http://$ROUTER_IP:8080$target_path" >/dev/null 2>&1
  fi
}
update_route_integration "POST /webhook/telegram" "/webhook/telegram"
update_route_integration "POST /internal/{proxy+}" "/internal/{proxy}"
update_route_integration "GET /health" "/health"
echo "  API Gateway -> http://$ROUTER_IP:8080 (webhook, internal, health)"

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
        headers={"Authorization": "Bearer $SMOKE_TOKEN"})
    r = urllib.request.urlopen(req, timeout=15)
    assert r.status == 200, f"status {r.status}"
check("GET /api/kpis", hit_dashboard)
if ok:
    print(f"SMOKE OK — API: {api}")
else:
    print("SMOKE COM FALHAS")
    raise SystemExit(1)
PYEOF

echo
echo "Deploy concluído."
echo "API (Gateway):       $API_URL"
echo "Dashboard:           $DASHBOARD_URL"
echo "Conversation Router: task do ECS sdr-conversation-router (porta 8080; escala 09:00-18:00 BRT)"
echo "Voice adapter:       task do ECS sdr-voice-adapter (faster-whisper; escala 09:00-18:00 BRT; consome sdr-voice-queue)"
echo "Token Telegram:      substitua em Secrets Manager (sdr/tg-bot-token) e re-aplique p/ ativar o bot"
echo "LLM (OpenRouter):    defina LLM_API_KEY em secrets.local.env e re-aplique — Terraform grava na secret sdr/llm-api-key"
echo "Teardown:            ./stop.sh"