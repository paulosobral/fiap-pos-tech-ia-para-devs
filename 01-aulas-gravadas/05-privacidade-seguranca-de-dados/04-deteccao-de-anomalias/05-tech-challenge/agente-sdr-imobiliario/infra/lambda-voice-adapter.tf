# voice-adapter agora roda em ECS Fargate (whisper real não cabe em Lambda zip).
# A fila SQS sdr-voice-queue é consumida pelo worker ECS (service/sqs_worker.py).
# A Lambda voice-adapter permanece para testes/unitários, mas o processamento
# de áudio em produção é feito pelo container ECS com faster-whisper.

module "lambda_voice_adapter" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-voice-adapter"
  description            = "u2 - transcricao de voz (whisper) — deprecated em producao, uso via ECS"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/voice-adapter.zip"
  timeout                = 300
  memory_size            = 2048
  ephemeral_storage_size = 1024

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    TELEGRAM_BOT_TOKEN    = var.telegram_bot_token
    INTERNAL_SECRET_TOKEN = aws_secretsmanager_secret_version.internal_secret_token.secret_string
    ROUTER_BASE_URL       = aws_apigatewayv2_api.http.api_endpoint
    SESSIONS_TABLE        = aws_dynamodb_table.sessions.name
  }
}