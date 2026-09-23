data "aws_caller_identity" "current" {}

resource "aws_apigatewayv2_api" "http" {
  name          = "sdr-http-api"
  protocol_type = "HTTP"
  description   = "API HTTP do Agente SDR (webhook, internals, health, dashboard)"
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.http.id
  name        = "$default"
  auto_deploy = true
}

# conversation-router agora em ECS Fargate.
# No POC sem ALB (custo mínimo e zero-idle), os endpoints são atendidos diretamente pelo container
# ou roteados via HTTP integration.
# Para manter a URL única do API Gateway para webhooks e health, mantemos a rota HTTP integration se desejado,
# ou expomos o endpoint do router.

# Integração HTTP_PROXY para o conversation-router (ECS Fargate).
# O URL aponta para o IP público da task; start.sh atualiza após o scale-out.
resource "aws_apigatewayv2_integration" "router" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "HTTP_PROXY"
  integration_uri        = var.router_endpoint
  integration_method     = "POST"
  payload_format_version = "1.0"
}

resource "aws_apigatewayv2_integration" "router_get" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "HTTP_PROXY"
  integration_uri        = var.router_endpoint
  integration_method     = "GET"
  payload_format_version = "1.0"
}

resource "aws_apigatewayv2_integration" "dashboard" {
  api_id                 = aws_apigatewayv2_api.http.id
  integration_type       = "AWS_PROXY"
  integration_uri        = module.lambda_dashboard_api.lambda_function_invoke_arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "dashboard_proxy" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /api/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.dashboard.id}"
}

# Rotas do conversation-router (HTTP_PROXY -> ECS task)
resource "aws_apigatewayv2_route" "webhook_telegram" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /webhook/telegram"
  target    = "integrations/${aws_apigatewayv2_integration.router.id}"
}

resource "aws_apigatewayv2_route" "internal_proxy" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "POST /internal/{proxy+}"
  target    = "integrations/${aws_apigatewayv2_integration.router.id}"
}

resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.http.id
  route_key = "GET /health"
  target    = "integrations/${aws_apigatewayv2_integration.router_get.id}"
}

resource "aws_lambda_permission" "dashboard_apigw" {
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_dashboard_api.lambda_function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http.execution_arn}/*/*/*"
}