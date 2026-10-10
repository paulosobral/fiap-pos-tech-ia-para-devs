resource "random_password" "internal_secret" {
  length  = 32
  special = false
  upper   = true
  lower   = true
  numeric = true
}

resource "aws_kms_key" "pii" {
  description             = "Chave de criptografia de PII (NFR5.x) - agente SDR POC"
  deletion_window_in_days = 7
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnableRootAdmin"
        Effect    = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action    = "kms:*"
        Resource  = "*"
      },
      {
        Sid       = "EnableLambdaUse"
        Effect    = "Allow"
        Principal = { Service = "lambda.amazonaws.com" }
        Action    = ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"]
        Resource  = "*"
      },
    ]
  })
}

resource "aws_kms_alias" "pii" {
  name          = "alias/${var.name_prefix}-pii"
  target_key_id = aws_kms_key.pii.key_id
}

resource "aws_secretsmanager_secret" "telegram_bot_token" {
  name                    = "sdr/tg-bot-token"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "telegram_bot_token" {
  count         = var.telegram_bot_token == "" ? 0 : 1
  secret_id     = aws_secretsmanager_secret.telegram_bot_token.id
  secret_string = var.telegram_bot_token
}

resource "aws_secretsmanager_secret" "llm_api_key" {
  name                    = "sdr/llm-api-key"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "llm_api_key" {
  count         = var.llm_api_key == "" ? 0 : 1
  secret_id     = aws_secretsmanager_secret.llm_api_key.id
  secret_string = var.llm_api_key
}

resource "aws_secretsmanager_secret" "internal_secret_token" {
  name                    = "sdr/dashboard-api-token"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "internal_secret_token" {
  secret_id     = aws_secretsmanager_secret.internal_secret_token.id
  secret_string = random_password.internal_secret.result
}

resource "aws_ssm_parameter" "llm_model" {
  name        = "/sdr/llm-model-primary"
  description = "Modelo primário LLM (Tier 1 — rotina/econômico: triagem, qualificação, follow-up) no OpenRouter"
  type        = "String"
  value       = var.llm_model
  overwrite   = true
}

resource "aws_ssm_parameter" "llm_model_fallback" {
  name        = "/sdr/llm-model-fallback"
  description = "Modelo de fallback automático (Tier 2 — usado pelo LiteLLM em 429/timeout/indisponibilidade do primário) no OpenRouter"
  type        = "String"
  value       = var.llm_model_fallback
  overwrite   = true
}

resource "aws_ssm_parameter" "llm_model_complex" {
  name        = "/sdr/llm-model-complex"
  description = "Modelo premium (Tier 3 — negociação sofisticada, dúvidas jurídicas, handoff executivo) no OpenRouter"
  type        = "String"
  value       = var.llm_model_complex
  overwrite   = true
}

locals {
  hubspot_enabled = var.hubspot_mcp_client_id != "" && var.hubspot_mcp_client_secret != "" && var.hubspot_mcp_refresh_token != ""
}

resource "aws_secretsmanager_secret" "hubspot_mcp" {
  name                    = "sdr/hubspot-mcp"
  recovery_window_in_days = 0
}
resource "aws_secretsmanager_secret_version" "hubspot_mcp" {
  count     = local.hubspot_enabled ? 1 : 0
  secret_id = aws_secretsmanager_secret.hubspot_mcp.id
  secret_string = jsonencode({
    client_id     = var.hubspot_mcp_client_id
    client_secret = var.hubspot_mcp_client_secret
    refresh_token = var.hubspot_mcp_refresh_token
  })
  # O refresh token roda a cada renovação (uso único): applies seguintes não podem sobrescrever o atual.
  lifecycle {
    ignore_changes = [secret_string]
  }
}

resource "aws_ssm_parameter" "bot_name" {
  name        = "/sdr/bot-name"
  description = "Nome da assistente: se apresenta assim na primeira mensagem (lido pelo conversation-router com cache de 5 min)"
  type        = "String"
  value       = var.bot_name
  overwrite   = true
}
