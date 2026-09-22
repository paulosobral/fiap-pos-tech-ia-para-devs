variable "region" {
  description = "Região AWS do POC (Q1 do environment-provisioning)"
  type        = string
  default     = "us-east-1"
}

variable "name_prefix" {
  description = "Prefixo dos recursos"
  type        = string
  default     = "sdr"
}

variable "telegram_bot_token" {
  description = "Token do bot do Telegram (SecureString). Vazio = telegram desligado até o humano preencher a secret e re-applicar."
  type        = string
  default     = ""
  sensitive   = true
}

variable "llm_api_key" {
  description = "Chave OpenRouter p/ o LLM do conversation-router (FR-02). Vazio = classificador por regex (POC sem IA)."
  type        = string
  default     = ""
  sensitive   = true
}

variable "dashboard_ui_image" {
  description = "URI da imagem ECR do dashboard-ui (build: podman build & push antes do apply). Default = pública python p/ permitir o 1º apply; o start.sh faz o apply2 com a imagem ECR."
  type        = string
  default     = "public.ecr.aws/docker/library/python:3.11-slim"
}

variable "dashboard_schedule_start" {
  description = "Cron (UTC) escala o ECS p/ 1 task (janela diária p/ Streamlit). Default 12:00 UTC = 09:00 BRT"
  type        = string
  default     = "cron(0 12 * * ? *)"
}

variable "dashboard_schedule_end" {
  description = "Cron (UTC) escala o ECS p/ 0 tasks (fora da janela). Default 21:00 UTC = 18:00 BRT"
  type        = string
  default     = "cron(0 21 * * ? *)"
}

variable "anomaly_schedule" {
  description = "Frequência do varrimento de anomalias (u5)"
  type        = string
  default     = "rate(1 minute)"
}

variable "voice_adapter_image" {
  description = "URI da imagem ECR do voice-adapter (build: podman build & push antes do apply). Default = python p/ permitir o 1º apply; o start.sh faz o apply2 com a imagem ECR."
  type        = string
  default     = "public.ecr.aws/docker/library/python:3.11-slim"
}

variable "voice_schedule_start" {
  description = "Cron (UTC) escala o ECS voice-adapter p/ 1 task. Default 12:00 UTC = 09:00 BRT"
  type        = string
  default     = "cron(0 12 * * ? *)"
}

variable "voice_schedule_end" {
  description = "Cron (UTC) escala o ECS voice-adapter p/ 0 tasks. Default 21:00 UTC = 18:00 BRT"
  type        = string
  default     = "cron(0 21 * * ? *)"
}

variable "followup_schedule" {
  description = "Frequência do follow-up agendado (u6) — cron com hora fixa UTC DENTRO da janela de silêncio 8-18 BRT (12:00 UTC = 09:00 BRT). No dev do rate(1 day), o tick cai fora da janela e o follow-up nunca sai."
  type        = string
  default     = "cron(0 12 * * ? *)"
}