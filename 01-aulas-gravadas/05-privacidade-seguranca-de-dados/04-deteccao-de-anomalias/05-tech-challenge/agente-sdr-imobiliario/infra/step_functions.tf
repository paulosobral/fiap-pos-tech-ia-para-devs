# Step Functions State Machine para Follow-up (PRD §8.3 / FR-06)
# Orquestra os follow-ups com wait states de 2h e 24h conforme PRD.

resource "aws_sfn_state_machine" "followup_pipeline" {
  name     = "${var.name_prefix}-followup-pipeline"
  role_arn = aws_iam_role.step_functions.arn

  definition = jsonencode({
    Comment = "Pipeline de Follow-up com Wait States (PRD §8.3)"
    StartAt = "WaitFirstFollowup"
    States = {
      WaitFirstFollowup = {
        Type    = "Wait"
        Seconds = 7200 # 2 horas (PRD §8.3)
        Next    = "SendFirstFollowup"
      }
      SendFirstFollowup = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = module.lambda_followup.lambda_function_arn
          Payload = {
            "lead_id.$" = "$.lead_id"
            "step"      = "followup_2h"
          }
        }
        ResultPath = "$.first_result"
        Next       = "CheckEngagement"
      }
      CheckEngagement = {
        Type = "Choice"
        Choices = [
          {
            Variable      = "$.first_result.Payload.engaged"
            BooleanEquals = true
            Next          = "Complete"
          }
        ]
        Default = "WaitSecondFollowup"
      }
      WaitSecondFollowup = {
        Type    = "Wait"
        Seconds = 86400 # 24 horas (PRD §8.3)
        Next    = "SendSecondFollowup"
      }
      SendSecondFollowup = {
        Type     = "Task"
        Resource = "arn:aws:states:::lambda:invoke"
        Parameters = {
          FunctionName = module.lambda_followup.lambda_function_arn
          Payload = {
            "lead_id.$" = "$.lead_id"
            "step"      = "followup_24h"
          }
        }
        ResultPath = "$.second_result"
        Next       = "Complete"
      }
      Complete = {
        Type = "Succeed"
      }
    }
  })

  tags = {
    Project   = var.name_prefix
    ManagedBy = "terraform"
  }
}

resource "aws_iam_role" "step_functions" {
  name = "${var.name_prefix}-sfn-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "states.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "step_functions_lambda" {
  name = "${var.name_prefix}-sfn-lambda-policy"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = [
          module.lambda_followup.lambda_function_arn
        ]
      }
    ]
  })
}
