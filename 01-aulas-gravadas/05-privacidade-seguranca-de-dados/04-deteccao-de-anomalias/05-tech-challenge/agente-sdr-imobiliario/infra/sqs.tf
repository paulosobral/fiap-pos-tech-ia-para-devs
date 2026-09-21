resource "aws_sqs_queue" "voice" {
  name                       = "sdr-voice-queue"
  delay_seconds              = 0
  message_retention_seconds  = 345600
  visibility_timeout_seconds = 300
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.voice_dlq.arn
    maxReceiveCount     = 3
  })
}
resource "aws_sqs_queue" "voice_dlq" {
  name                      = "sdr-voice-queue-dlq"
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "crm" {
  name                       = "sdr-crm-queue"
  delay_seconds              = 0
  message_retention_seconds  = 345600
  visibility_timeout_seconds = 300
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.crm_dlq.arn
    maxReceiveCount     = 3
  })
}
resource "aws_sqs_queue" "crm_dlq" {
  name                      = "sdr-crm-queue-dlq"
  message_retention_seconds = 1209600
}

resource "aws_sqs_queue" "ingest" {
  name                       = "sdr-ingest-queue"
  delay_seconds              = 0
  message_retention_seconds  = 345600
  visibility_timeout_seconds = 300
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingest_dlq.arn
    maxReceiveCount     = 3
  })
}
resource "aws_sqs_queue" "ingest_dlq" {
  name                      = "sdr-ingest-queue-dlq"
  message_retention_seconds = 1209600
}