module "lambda_crm_adapter" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-crm-adapter"
  description            = "u3 - sincronizacao com CRM (filas)"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/crm-adapter.zip"
  timeout                = 60
  memory_size            = 256

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    INTERNAL_SECRET_TOKEN = aws_secretsmanager_secret_version.internal_secret_token.secret_string
    FLOW_BASE_URL         = aws_apigatewayv2_api.http.api_endpoint
    SESSIONS_TABLE        = aws_dynamodb_table.sessions.name
    CRM_MAX_RECEIVES      = "5"
  }
}

resource "aws_lambda_event_source_mapping" "crm_from_sqs" {
  event_source_arn = aws_sqs_queue.crm.arn
  function_name    = module.lambda_crm_adapter.lambda_function_arn
  batch_size       = 5
}

resource "aws_lambda_permission" "crm_sqs" {
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_crm_adapter.lambda_function_name
  principal     = "sqs.amazonaws.com"
  source_arn    = aws_sqs_queue.crm.arn
}