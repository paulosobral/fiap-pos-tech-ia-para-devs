resource "aws_dynamodb_table" "sessions" {
  name         = "sdr-sessions"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "telegram_user_id"
  range_key    = "started_at"
  attribute {
    name = "telegram_user_id"
    type = "S"
  }
  attribute {
    name = "started_at"
    type = "S"
  }
  attribute {
    name = "lead_status"
    type = "S"
  }
  global_secondary_index {
    name            = "lead-index"
    hash_key        = "lead_status"
    range_key       = "started_at"
    projection_type = "ALL"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "pii" {
  name         = "sdr-pii"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "telegram_user_id"
  attribute {
    name = "telegram_user_id"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "alerts" {
  name         = "sdr-alerts"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "alert_id"
  attribute {
    name = "alert_id"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "dedupe" {
  name         = "sdr-ingest-dedupe"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "dedupe_key"
  attribute {
    name = "dedupe_key"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}

resource "aws_dynamodb_table" "followup" {
  name         = "sdr-followup-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "telegram_user_id"
  attribute {
    name = "telegram_user_id"
    type = "S"
  }
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }
}