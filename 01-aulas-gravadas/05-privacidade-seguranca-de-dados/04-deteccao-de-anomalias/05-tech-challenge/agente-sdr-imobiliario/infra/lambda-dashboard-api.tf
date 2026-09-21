module "lambda_dashboard_api" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-dashboard-api"
  description            = "u7-api - API de KPIs do dashboard (HTTP)"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/dashboard-api.zip"
  timeout                = 30
  memory_size            = 256

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    SESSIONS_TABLE           = aws_dynamodb_table.sessions.name
    ALERTS_TABLE             = aws_dynamodb_table.alerts.name
    CW_NAMESPACE             = "SDR/AgenteImobiliario"
    CW_RESPONSE_METRIC       = "ResponseMs"
    CW_COST_METRIC           = "EstimatedCostUsd"
    DASHBOARD_ALLOWED_ORIGIN = "*"
  }
}