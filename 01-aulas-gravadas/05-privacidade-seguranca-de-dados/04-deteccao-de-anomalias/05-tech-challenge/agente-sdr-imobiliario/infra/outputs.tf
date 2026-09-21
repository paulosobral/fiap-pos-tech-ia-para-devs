output "api_url" {
  value = aws_apigatewayv2_api.http.api_endpoint
}

output "region" {
  value = var.region
}

output "dashboard_ui" {
  value = "endereço do Streamlit obtido via: DELAYED (task IP pós apply2)"
}

output "secrets_manager_telegram" {
  value = aws_secretsmanager_secret.telegram_bot_token.name
}