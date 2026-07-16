# ==============================================================================
# S3 Bucket — armazenamento de entrada/saída do AWS Transcribe (aba Áudio)
#
# O job assíncrono do Transcribe exige um bucket S3. A aplicação lê o nome
# do bucket da env AUDIO_TRANSCRIBE_BUCKET (ver audio/aws_speech.py).
# ==============================================================================

resource "aws_s3_bucket" "transcribe_bucket" {
  bucket        = var.transcribe_bucket_name
  force_destroy = var.s3_force_destroy
}

resource "aws_s3_bucket_server_side_encryption_configuration" "transcribe_bucket_encryption" {
  bucket = aws_s3_bucket.transcribe_bucket.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "transcribe_bucket_public_access" {
  bucket = aws_s3_bucket.transcribe_bucket.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
