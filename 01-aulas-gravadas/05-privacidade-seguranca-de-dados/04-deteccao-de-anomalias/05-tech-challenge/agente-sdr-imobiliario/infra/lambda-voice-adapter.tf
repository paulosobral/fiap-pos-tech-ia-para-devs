module "lambda_voice_adapter" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-voice-adapter"
  description            = "u2 - transcricao de voz (whisper) e envio ao router"
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

resource "aws_lambda_event_source_mapping" "voice_from_sqs" {
  event_source_arn = aws_sqs_queue.voice.arn
  function_name    = module.lambda_voice_adapter.lambda_function_arn
  batch_size       = 5
}

resource "aws_lambda_permission" "voice_sqs" {
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_voice_adapter.lambda_function_name
  principal     = "sqs.amazonaws.com"
  source_arn    = aws_sqs_queue.voice.arn
}