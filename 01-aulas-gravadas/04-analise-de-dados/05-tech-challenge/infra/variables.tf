# ==============================================================================
# Variáveis
# ==============================================================================

variable "aws_region" {
  description = "Região AWS onde o bucket será criado. Deve casar com a região do AWS Transcribe usado pela aba Áudio (audio/aws_speech.py usa us-east-1 fixo)."
  type        = string
  default     = "us-east-1"
}

# ==============================================================================
# S3 — bucket de transcrição (aba Áudio)
# ==============================================================================

variable "transcribe_bucket_name" {
  description = "Nome do bucket S3 usado pelo AWS Transcribe (entrada/saída do job assíncrono). Corresponde à env AUDIO_TRANSCRIBE_BUCKET."
  type        = string
  default     = "psobral89-bucket-transcribe"
}

variable "s3_force_destroy" {
  description = "Permitir destruir o bucket S3 mesmo com objetos dentro"
  type        = bool
  default     = true
}
