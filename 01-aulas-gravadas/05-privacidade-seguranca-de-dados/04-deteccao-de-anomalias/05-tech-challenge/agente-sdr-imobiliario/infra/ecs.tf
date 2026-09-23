# ECR do dashboard-ui (único container do POC — decisão do humano no deployment-execution)
resource "aws_ecr_repository" "dashboard_ui" {
  name                 = "sdr-dashboard-ui"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

data "aws_caller_identity" "ecr_current" {}

output "dashboard_ecr_repo" {
  value = aws_ecr_repository.dashboard_ui.repository_url
}

# Cluster
resource "aws_ecs_cluster" "sdr" {
  name = "sdr-cluster"
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default_public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "map-public-ip-on-launch"
    values = ["true"]
  }
}

resource "aws_security_group" "dashboard_ui" {
  name        = "sdr-dashboard-ui"
  description = "Acesso ao Streamlit do dashboard-ui (IP publico da task)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "dashboard_ui" {
  family                   = "sdr-dashboard-ui"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "256"
  memory                   = "512"
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([
    {
      name         = "dashboard-ui"
      image        = var.dashboard_ui_image
      essential    = true
      portMappings = [{ containerPort = 80, protocol = "tcp" }]
      environment = [
        { name = "DASHBOARD_API_URL", value = aws_apigatewayv2_api.http.api_endpoint },
        { name = "DASHBOARD_API_TOKEN", value = random_password.internal_secret.result },
        { name = "DASHBOARD_ALLOWED_ORIGIN", value = "*" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/sdr-dashboard-ui"
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "dashboard-ui"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "dashboard_ui" {
  name            = "sdr-dashboard-ui"
  cluster         = aws_ecs_cluster.sdr.id
  task_definition = aws_ecs_task_definition.dashboard_ui.arn
  desired_count   = 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default_public.ids
    security_groups  = [aws_security_group.dashboard_ui.id]
    assign_public_ip = true
  }
}

# Scale-to-zero fora da janela (decisão do humano: 8h/dia)
resource "aws_appautoscaling_target" "dashboard_ui" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.sdr.name}/${aws_ecs_service.dashboard_ui.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = 0
  max_capacity       = 1
}

resource "aws_appautoscaling_scheduled_action" "scale_out" {
  name               = "sdr-dashboard-start"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.dashboard_ui.resource_id
  scalable_dimension = aws_appautoscaling_target.dashboard_ui.scalable_dimension
  schedule           = var.dashboard_schedule_start
  timezone           = "UTC"
  scalable_target_action {
    min_capacity = 1
    max_capacity = 1
  }
}

resource "aws_appautoscaling_scheduled_action" "scale_in" {
  name               = "sdr-dashboard-stop"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.dashboard_ui.resource_id
  scalable_dimension = aws_appautoscaling_target.dashboard_ui.scalable_dimension
  schedule           = var.dashboard_schedule_end
  timezone           = "UTC"
  scalable_target_action {
    min_capacity = 0
    max_capacity = 0
  }
}

resource "aws_cloudwatch_log_group" "dashboard_ui" {
  name              = "/ecs/sdr-dashboard-ui"
  retention_in_days = 7
}

resource "aws_iam_role" "ecs_execution" {
  name = "sdr-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "ecs_execution" {
  name = "sdr-ecs-execution"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ECR"
        Effect   = "Allow"
        Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"]
        Resource = aws_ecr_repository.dashboard_ui.arn
      },
      {
        Sid      = "ECRAuth"
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Sid      = "Logs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.dashboard_ui.arn}:*"
      },
    ]
  })
}

output "dashboard_ui_ip" {
  value = "IP da task: ver 'aws ecs describe-tasks' ou o log do start.sh (sem ALB — acesso direto)"
}

output "router_ip" {
  value = "IP da task: ver 'aws ecs describe-tasks' ou o log do start.sh (sem ALB — porta 8080)"
}

# ============================================================
# Voice Adapter — ECS Fargate (whisper real, não cabe em Lambda)
# ============================================================

resource "aws_ecr_repository" "voice_adapter" {
  name                 = "sdr-voice-adapter"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

output "voice_ecr_repo" {
  value = aws_ecr_repository.voice_adapter.repository_url
}

resource "aws_cloudwatch_log_group" "voice_adapter" {
  name              = "/ecs/sdr-voice-adapter"
  retention_in_days = 7
}

resource "aws_security_group" "voice_adapter" {
  name        = "sdr-voice-adapter"
  description = "Voice adapter worker (sem ingress, consome SQS)"
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "voice_adapter" {
  family                   = "sdr-voice-adapter"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "1024"
  memory                   = "4096"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.voice_adapter_task.arn

  container_definitions = jsonencode([
    {
      name      = "voice-adapter"
      image     = var.voice_adapter_image
      essential = true
      environment = [
        { name = "VOICE_QUEUE_URL", value = aws_sqs_queue.voice.url },
        { name = "TELEGRAM_BOT_TOKEN", value = var.telegram_bot_token },
        { name = "INTERNAL_SECRET_TOKEN", value = random_password.internal_secret.result },
        { name = "ROUTER_BASE_URL", value = aws_apigatewayv2_api.http.api_endpoint },
        { name = "SESSIONS_TABLE", value = aws_dynamodb_table.sessions.name },
        { name = "AWS_REGION", value = var.region },
        { name = "WORKER_POLL_INTERVAL", value = "5" },
        { name = "WORKER_MAX_MESSAGES", value = "5" },
        { name = "WHISPER_MODEL_SIZE", value = "small" },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/sdr-voice-adapter"
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "voice-adapter"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "voice_adapter" {
  name            = "sdr-voice-adapter"
  cluster         = aws_ecs_cluster.sdr.id
  task_definition = aws_ecs_task_definition.voice_adapter.arn
  desired_count   = 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default_public.ids
    security_groups  = [aws_security_group.voice_adapter.id]
    assign_public_ip = true
  }
}

# Scale-to-zero fora da janela comercial (09:00-18:00 BRT)
resource "aws_appautoscaling_target" "voice_adapter" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.sdr.name}/${aws_ecs_service.voice_adapter.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = 0
  max_capacity       = 1
}

resource "aws_appautoscaling_scheduled_action" "voice_scale_out" {
  name               = "sdr-voice-start"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.voice_adapter.resource_id
  scalable_dimension = aws_appautoscaling_target.voice_adapter.scalable_dimension
  schedule           = var.voice_schedule_start
  timezone           = "UTC"
  scalable_target_action {
    min_capacity = 1
    max_capacity = 1
  }
}

resource "aws_appautoscaling_scheduled_action" "voice_scale_in" {
  name               = "sdr-voice-stop"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.voice_adapter.resource_id
  scalable_dimension = aws_appautoscaling_target.voice_adapter.scalable_dimension
  schedule           = var.voice_schedule_end
  timezone           = "UTC"
  scalable_target_action {
    min_capacity = 0
    max_capacity = 0
  }
}

# IAM role da task (permite ler SQS, DynamoDB, KMS, Secrets)
resource "aws_iam_role" "voice_adapter_task" {
  name = "sdr-voice-adapter-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy" "voice_adapter_task" {
  name = "sdr-voice-adapter-task"
  role = aws_iam_role.voice_adapter_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "SQS"
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes",
          "sqs:GetQueueUrl", "sqs:ChangeMessageVisibility"
        ]
        Resource = aws_sqs_queue.voice.arn
      },
      {
        Sid    = "DynamoDB"
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem", "dynamodb:Query"
        ]
        Resource = [
          "arn:aws:dynamodb:${var.region}:*:table/sdr-*",
          "arn:aws:dynamodb:${var.region}:*:table/sdr-*/index/*",
        ]
      },
      {
        Sid      = "KMS"
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
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
    ]
  })
}

# ============================================================
# Conversation Router — ECS Fargate (litellm + langgraph + faiss completos)
# ============================================================

resource "aws_ecr_repository" "conversation_router" {
  name                 = "sdr-conversation-router"
  image_tag_mutability = "MUTABLE"
  force_delete         = true
  image_scanning_configuration {
    scan_on_push = true
  }
}

output "router_ecr_repo" {
  value = aws_ecr_repository.conversation_router.repository_url
}

resource "aws_cloudwatch_log_group" "conversation_router" {
  name              = "/ecs/sdr-conversation-router"
  retention_in_days = 7
}

resource "aws_security_group" "conversation_router" {
  name        = "sdr-conversation-router"
  description = "Conversation router container (HTTP 8080 do API Gateway e interno)"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_ecs_task_definition" "conversation_router" {
  family                   = "sdr-conversation-router"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.router_task.arn

  container_definitions = jsonencode([
    {
      name         = "conversation-router"
      image        = var.router_image
      essential    = true
      portMappings = [{ containerPort = 8080, protocol = "tcp" }]
      environment = [
        { name = "PORT", value = "8080" },
        { name = "TELEGRAM_BOT_TOKEN", value = var.telegram_bot_token },
        { name = "TELEGRAM_SECRET_TOKEN", value = random_password.internal_secret.result },
        { name = "INTERNAL_SECRET_TOKEN", value = random_password.internal_secret.result },
        { name = "LLM_MODEL", value = var.llm_model },
        { name = "LLM_MODEL_SSM_PARAM", value = aws_ssm_parameter.llm_model.name },
        { name = "LLM_API_SECRET_ID", value = aws_secretsmanager_secret.llm_api_key.arn },
        { name = "PII_KMS_KEY_ID", value = aws_kms_key.pii.key_id },
        { name = "SESSIONS_TABLE", value = aws_dynamodb_table.sessions.name },
        { name = "PII_TABLE", value = aws_dynamodb_table.pii.name },
        { name = "ALERTS_TABLE", value = aws_dynamodb_table.alerts.name },
        { name = "VOICE_QUEUE_URL", value = aws_sqs_queue.voice.id },
        { name = "CRM_QUEUE_URL", value = aws_sqs_queue.crm.id },
        { name = "CATALOG_BUCKET", value = aws_s3_bucket.catalogs.bucket },
        { name = "SPECIALIST_ROTATION", value = "Adriana, Bruno, Carla" },
        { name = "SPECIALIST_FALLBACK", value = "diretor" },
        { name = "AWS_REGION", value = var.region },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = "/ecs/sdr-conversation-router"
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = "conversation-router"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "conversation_router" {
  name            = "sdr-conversation-router"
  cluster         = aws_ecs_cluster.sdr.id
  task_definition = aws_ecs_task_definition.conversation_router.arn
  desired_count   = 0
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = data.aws_subnets.default_public.ids
    security_groups  = [aws_security_group.conversation_router.id]
    assign_public_ip = true
  }
}

# Scale-to-zero fora da janela comercial (09:00-18:00 BRT)
resource "aws_appautoscaling_target" "conversation_router" {
  service_namespace  = "ecs"
  resource_id        = "service/${aws_ecs_cluster.sdr.name}/${aws_ecs_service.conversation_router.name}"
  scalable_dimension = "ecs:service:DesiredCount"
  min_capacity       = 0
  max_capacity       = 1
}

resource "aws_appautoscaling_scheduled_action" "router_scale_out" {
  name               = "sdr-router-start"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.conversation_router.resource_id
  scalable_dimension = aws_appautoscaling_target.conversation_router.scalable_dimension
  schedule           = var.router_schedule_start
  timezone           = "UTC"
  scalable_target_action {
    min_capacity = 1
    max_capacity = 1
  }
}

resource "aws_appautoscaling_scheduled_action" "router_scale_in" {
  name               = "sdr-router-stop"
  service_namespace  = "ecs"
  resource_id        = aws_appautoscaling_target.conversation_router.resource_id
  scalable_dimension = aws_appautoscaling_target.conversation_router.scalable_dimension
  schedule           = var.router_schedule_end
  timezone           = "UTC"
  scalable_target_action {
    min_capacity = 0
    max_capacity = 0
  }
}

# IAM role da task do conversation-router (reutiliza permissões completas de DynamoDB, SQS, KMS, Secrets, S3)
resource "aws_iam_role" "router_task" {
  name = "sdr-conversation-router-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action    = "sts:AssumeRole"
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "router_task_policy" {
  role       = aws_iam_role.router_task.name
  policy_arn = aws_iam_policy.sdr_lambda.arn
}

resource "aws_iam_role_policy" "ecs_execution_router_ecr" {
  name = "sdr-ecs-execution-router-ecr"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "RouterECR"
        Effect   = "Allow"
        Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"]
        Resource = aws_ecr_repository.conversation_router.arn
      },
      {
        Sid      = "RouterLogs"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.conversation_router.arn}:*"
      }
    ]
  })
}

# Permissão de ECR para a role de execução puxar a imagem do voice-adapter
resource "aws_iam_role_policy" "ecs_execution_voice" {
  name = "sdr-ecs-execution-voice"
  role = aws_iam_role.ecs_execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "ECRVoice"
        Effect   = "Allow"
        Action   = ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability"]
        Resource = aws_ecr_repository.voice_adapter.arn
      },
      {
        Sid      = "LogsVoice"
        Effect   = "Allow"
        Action   = ["logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "${aws_cloudwatch_log_group.voice_adapter.arn}:*"
      },
    ]
  })
}