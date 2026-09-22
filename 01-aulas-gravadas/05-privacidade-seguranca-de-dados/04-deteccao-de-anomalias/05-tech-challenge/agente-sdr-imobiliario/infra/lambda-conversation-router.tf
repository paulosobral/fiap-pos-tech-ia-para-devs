module "lambda_conversation_router" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-conversation-router"
  description            = "u1 - roteador de conversa (webhook Telegram + internals)"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/conversation-router.zip"
  timeout                = 60
  memory_size            = 256

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    TELEGRAM_BOT_TOKEN    = var.telegram_bot_token
    INTERNAL_SECRET_TOKEN = aws_secretsmanager_secret_version.internal_secret_token.secret_string
    LLM_API_KEY           = var.llm_api_key
    PII_KMS_KEY_ID        = aws_kms_key.pii.key_id
    SESSIONS_TABLE        = aws_dynamodb_table.sessions.name
    PII_TABLE             = aws_dynamodb_table.pii.name
    ALERTS_TABLE          = aws_dynamodb_table.alerts.name
    VOICE_QUEUE_URL       = aws_sqs_queue.voice.id
    CRM_QUEUE_URL         = aws_sqs_queue.crm.id
    SPECIALIST_ROTATION   = "Adriana, Bruno, Carla"
    SPECIALIST_FALLBACK   = "diretor"
  }
}