module "lambda_followup" {
  source  = "terraform-aws-modules/lambda/aws"
  version = "~> 7.0"

  function_name          = "sdr-followup"
  description            = "u6 - follow-up agendado (EventBridge schedule)"
  handler                = "handler.handler"
  runtime                = "python3.11"
  create_package         = false
  local_existing_package = "../dist/followup.zip"
  timeout                = 60
  memory_size            = 256

  attach_policies    = true
  number_of_policies = 1
  policies           = [aws_iam_policy.sdr_lambda.arn]

  environment_variables = {
    TELEGRAM_BOT_TOKEN    = var.telegram_bot_token
    SESSIONS_TABLE        = aws_dynamodb_table.sessions.name
    FOLLOWUP_TABLE        = aws_dynamodb_table.followup.name
    FOLLOWUP_CADENCE_DAYS = "7"
  }
}

resource "aws_cloudwatch_event_rule" "followup" {
  name                = "sdr-followup-scan"
  schedule_expression = var.followup_schedule
}

resource "aws_cloudwatch_event_target" "followup" {
  rule      = aws_cloudwatch_event_rule.followup.name
  target_id = "sdr-followup-lambda"
  arn       = module.lambda_followup.lambda_function_arn
}

resource "aws_lambda_permission" "followup_events" {
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_followup.lambda_function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.followup.arn
}