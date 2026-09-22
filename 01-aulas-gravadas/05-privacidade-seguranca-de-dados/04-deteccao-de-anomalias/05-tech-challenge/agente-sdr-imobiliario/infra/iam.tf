resource "aws_iam_policy" "sdr_lambda" {
  name        = "sdr-lambda-policy"
  description = "Permissões mínimas das Lambdas (DynamoDB, SQS, KMS, Secrets, CloudWatch)"
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "DynamoDB"
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem", "dynamodb:Query", "dynamodb:Scan",
          "dynamodb:PutItem", "dynamodb:UpdateItem", "dynamodb:DeleteItem",
          "dynamodb:BatchGetItem", "dynamodb:BatchWriteItem",
        ]
        Resource = [
          "arn:aws:dynamodb:${var.region}:*:table/sdr-*",
          "arn:aws:dynamodb:${var.region}:*:table/sdr-*/index/*",
        ]
      },
      {
        Sid    = "SQS"
        Effect = "Allow"
        Action = [
          "sqs:SendMessage", "sqs:ReceiveMessage", "sqs:DeleteMessage",
          "sqs:GetQueueAttributes", "sqs:GetQueueUrl", "sqs:ChangeMessageVisibility",
        ]
        Resource = "arn:aws:sqs:${var.region}:*:sdr-*"
      },
      {
        Sid      = "KMS"
        Effect   = "Allow"
        Action   = ["kms:Encrypt", "kms:Decrypt", "kms:GenerateDataKey"]
        Resource = aws_kms_key.pii.arn
      },
      {
        Sid    = "SecretsManager"
        Effect = "Allow"
        Action = ["secretsmanager:GetSecretValue"]
        Resource = [
          aws_secretsmanager_secret.telegram_bot_token.arn,
          aws_secretsmanager_secret.internal_secret_token.arn,
        ]
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:${var.region}:*:*"
      },
      {
        Sid    = "S3Catalogs"
        Effect = "Allow"
        Action = [
          "s3:GetObject", "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.catalogs.arn,
          "${aws_s3_bucket.catalogs.arn}/*"
        ]
      },
    ]
  })
}