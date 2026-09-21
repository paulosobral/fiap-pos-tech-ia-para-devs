module "lambda_anomaly_detector" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-anomaly-detector"
  description            = "u5 - deteccao de anomalias (EventBridge schedule)"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/anomaly-detector.zip"
  timeout                = 60
  memory_size            = 256

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    SESSIONS_TABLE    = aws_dynamodb_table.sessions.name
    ALERTS_TABLE      = aws_dynamodb_table.alerts.name
    ANOMALY_SCORER    = "heuristic"
    ANOMALY_THRESHOLD = "0.7"
  }
}

resource "aws_cloudwatch_event_rule" "anomaly" {
  name                = "sdr-anomaly-scan"
  schedule_expression = var.anomaly_schedule
}

resource "aws_cloudwatch_event_target" "anomaly" {
  rule      = aws_cloudwatch_event_rule.anomaly.name
  target_id = "sdr-anomaly-lambda"
  arn       = module.lambda_anomaly_detector.lambda_function_arn
}

resource "aws_lambda_permission" "anomaly_events" {
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_anomaly_detector.lambda_function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.anomaly.arn
}