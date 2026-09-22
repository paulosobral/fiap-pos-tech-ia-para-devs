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