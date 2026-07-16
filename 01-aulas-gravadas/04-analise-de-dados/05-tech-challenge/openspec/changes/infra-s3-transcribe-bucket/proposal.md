## Why

A aba Áudio exige um bucket S3 (`AUDIO_TRANSCRIBE_BUCKET`) para o job assíncrono do AWS Transcribe, mas hoje o provisionamento é manual (`aws s3 mb ...`) e o `infra/` herdado do projeto `avc-stroke-prediction` referencia recursos inexistentes (SageMaker, IAM role, `local.bucket_name`), sem sequer um `provider` — não roda. Falta também um fluxo repetível de subir a app apontando pro bucket e derrubar tudo depois de uma demo.

## What Changes

- Reduzir `infra/` a apenas o necessário para o bucket S3 do Transcribe: `provider "aws"`, uma variável de nome de bucket, `s3_force_destroy`, `aws_region` (default `us-east-1`, casando com a região hardcoded em `audio/aws_speech.py`).
- Remover objetos/prefixos/recursos SageMaker herdados que não pertencem a este projeto.
- Adicionar bucket com server-side encryption (AES256) e public-access-block.
- Expor outputs `transcribe_bucket_name` e `transcribe_bucket_arn`.
- Adicionar `start.sh` na raiz: `terraform apply -auto-approve`, ativa o venv e sobe o Streamlit via `nohup` com `AUDIO_TRANSCRIBE_BUCKET` apontando pro bucket criado, imprimindo PID e URL.
- Adicionar `stop.sh` na raiz: mata o processo do Streamlit e roda `terraform destroy -auto-approve`.

## Capabilities

### New Capabilities
- `infra-transcribe-bucket`: provisionamento via Terraform do bucket S3 usado pela aba Áudio e scripts de ciclo de vida (subir app + apply / derrubar app + destroy).

### Modified Capabilities
<!-- Nenhuma. Não altera requisitos de capacidades existentes; a aba Áudio já consome AUDIO_TRANSCRIBE_BUCKET. -->

## Impact

- `infra/versions.tf`, `infra/variables.tf`, `infra/s3.tf`, `infra/outputs.tf` — reescritos/reduzidos.
- `start.sh`, `stop.sh` — novos scripts na raiz do projeto.
- Depende de: Terraform >= 1.5, provider AWS ~> 5.0, credenciais AWS com permissão de S3, `venv/` já instalado.
- Sem mudança de código Python; `audio/aws_speech.py` continua lendo a env `AUDIO_TRANSCRIBE_BUCKET`.
