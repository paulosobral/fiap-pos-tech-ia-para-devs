# Bucket S3 para catálogos do RAG (imóveis e clientes sintéticos)
# PRD §7.1 e §8.2: "Catálogos de imóveis e clientes armazenados no S3"

resource "aws_s3_bucket" "catalogs" {
  bucket_prefix = "${var.name_prefix}-catalogs-"
  force_destroy = true

  tags = {
    Project   = var.name_prefix
    ManagedBy = "terraform"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "catalogs" {
  bucket = aws_s3_bucket.catalogs.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "catalogs" {
  bucket = aws_s3_bucket.catalogs.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_object" "properties_catalog" {
  bucket       = aws_s3_bucket.catalogs.id
  key          = "catalogs/properties.json"
  source       = "${path.module}/../apps/conversation-router/data/properties.json"
  etag         = fileexists("${path.module}/../apps/conversation-router/data/properties.json") ? filemd5("${path.module}/../apps/conversation-router/data/properties.json") : null
  content_type = "application/json"
}

resource "aws_s3_object" "clients_catalog" {
  bucket       = aws_s3_bucket.catalogs.id
  key          = "catalogs/clients.json"
  source       = "${path.module}/../apps/conversation-router/data/clients.json"
  etag         = fileexists("${path.module}/../apps/conversation-router/data/clients.json") ? filemd5("${path.module}/../apps/conversation-router/data/clients.json") : null
  content_type = "application/json"
}
