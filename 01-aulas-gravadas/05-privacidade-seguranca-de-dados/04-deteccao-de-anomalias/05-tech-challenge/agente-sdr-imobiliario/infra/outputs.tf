output "api_url" {
  value = aws_apigatewayv2_api.http.api_endpoint
}

output "region" {
  value = var.region
}

output "dashboard_ui" {
  value = "http://<DASHBOARD_TASK_PUBLIC_IP> (obtido via start.sh)"
}

output "secrets_manager_telegram" {
  value = aws_secretsmanager_secret.telegram_bot_token.name
}