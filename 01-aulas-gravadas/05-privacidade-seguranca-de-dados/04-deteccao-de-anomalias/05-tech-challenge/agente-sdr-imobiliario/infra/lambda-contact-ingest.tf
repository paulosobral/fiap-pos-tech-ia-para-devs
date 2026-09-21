module "lambda_contact_ingest" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-contact-ingest"
  description            = "u4 - ingestao de contatos (fila + dedupe)"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/contact-ingest.zip"
  timeout                = 60
  memory_size            = 256

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    INTERNAL_SECRET_TOKEN = aws_secretsmanager_secret_version.internal_secret_token.secret_string
    ROUTER_BASE_URL       = aws_apigatewayv2_api.http.api_endpoint
    SESSIONS_TABLE        = aws_dynamodb_table.sessions.name
    DEDUPE_TABLE          = aws_dynamodb_table.dedupe.name
  }
}

resource "aws_lambda_event_source_mapping" "ingest_from_sqs" {
  event_source_arn = aws_sqs_queue.ingest.arn
  function_name    = module.lambda_contact_ingest.lambda_function_arn
  batch_size       = 5
}

resource "aws_lambda_permission" "ingest_sqs" {
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_contact_ingest.lambda_function_name
  principal     = "sqs.amazonaws.com"
  source_arn    = aws_sqs_queue.ingest.arn
}