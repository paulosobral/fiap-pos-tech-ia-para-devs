# Amazon SES Receipt Rule Set para Ingestão de Leads por Email (PRD §8.5 / FR-01)
# Encaminha emails recebidos para o Lambda contact-ingest via S3 ou SNS/Lambda direto.

resource "aws_ses_receipt_rule_set" "main" {
  rule_set_name = "${var.name_prefix}-ruleset"
}

resource "aws_ses_active_receipt_rule_set" "main" {
  rule_set_name = aws_ses_receipt_rule_set.main.rule_set_name
}

resource "aws_ses_receipt_rule" "leads" {
  name          = "${var.name_prefix}-leads-rule"
  rule_set_name = aws_ses_receipt_rule_set.main.rule_set_name
  recipients    = ["leads@${var.name_prefix}.local"]
  enabled       = true
  scan_enabled  = true

  lambda_action {
    function_arn    = module.lambda_contact_ingest.lambda_function_arn
    position        = 1
    invocation_type = "Event"
  }
}

resource "aws_lambda_permission" "allow_ses" {
  statement_id  = "AllowExecutionFromSES"
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_contact_ingest.lambda_function_name
  principal     = "ses.amazonaws.com"
}
