# ==============================================================================
# Outputs
# ==============================================================================

output "transcribe_bucket_name" {
  description = "Nome do bucket S3 do Transcribe (usar em AUDIO_TRANSCRIBE_BUCKET)"
  value       = aws_s3_bucket.transcribe_bucket.id
}

output "transcribe_bucket_arn" {
  description = "ARN do bucket S3 do Transcribe"
  value       = aws_s3_bucket.transcribe_bucket.arn
}
