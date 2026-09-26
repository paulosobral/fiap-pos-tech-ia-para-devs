# cloudwatch-logs.tf — Centralização de todos os CloudWatch Log Groups
# Garante que os grupos sejam criados antes das Lambdas/ECS e deletados depois,
# evitando ResourceAlreadyExistsException em ciclos de deploy/destroy.

locals {
  default_log_retention_days = 14
}

# -----------------------------------------------------------------------------
# Lambdas
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "lambda_anomaly_detector" {
  name              = "/aws/lambda/sdr-anomaly-detector"
  retention_in_days = local.default_log_retention_days
}

resource "aws_cloudwatch_log_group" "lambda_contact_ingest" {
  name              = "/aws/lambda/sdr-contact-ingest"
  retention_in_days = local.default_log_retention_days
}

resource "aws_cloudwatch_log_group" "lambda_crm_adapter" {
  name              = "/aws/lambda/sdr-crm-adapter"
  retention_in_days = local.default_log_retention_days
}

resource "aws_cloudwatch_log_group" "lambda_dashboard_api" {
  name              = "/aws/lambda/sdr-dashboard-api"
  retention_in_days = local.default_log_retention_days
}

resource "aws_cloudwatch_log_group" "lambda_followup" {
  name              = "/aws/lambda/sdr-followup"
  retention_in_days = local.default_log_retention_days
}

resource "aws_cloudwatch_log_group" "lambda_voice_adapter" {
  name              = "/aws/lambda/sdr-voice-adapter"
  retention_in_days = local.default_log_retention_days
}

# -----------------------------------------------------------------------------
# ECS Tasks
# -----------------------------------------------------------------------------
resource "aws_cloudwatch_log_group" "dashboard_ui" {
  name              = "/ecs/sdr-dashboard-ui"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "voice_adapter" {
  name              = "/ecs/sdr-voice-adapter"
  retention_in_days = 7
}

resource "aws_cloudwatch_log_group" "conversation_router" {
  name              = "/ecs/sdr-conversation-router"
  retention_in_days = 7
}
