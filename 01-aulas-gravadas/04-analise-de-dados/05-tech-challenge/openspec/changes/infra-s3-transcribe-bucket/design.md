## Context

O `infra/` foi copiado do projeto `avc-stroke-prediction` e trazia SageMaker notebook, IAM role, upload de dezenas de objetos e `local.bucket_name` — nenhum definido aqui e sem `provider`, então nem `plan` roda. A única necessidade real de infra deste projeto é o bucket S3 do AWS Transcribe consumido pela aba Áudio via env `AUDIO_TRANSCRIBE_BUCKET`. A região do Transcribe está hardcoded em `audio/aws_speech.py` (`AWS_REGION = "us-east-1"`).

## Goals / Non-Goals

**Goals:**
- `infra/` mínimo e autossuficiente que cria apenas o bucket S3, com `terraform validate` passando.
- Ciclo de vida de demo em um comando: `start.sh` (apply + app) e `stop.sh` (kill + destroy).
- Bucket seguro por padrão (encryption + public-access-block).

**Non-Goals:**
- SageMaker, IAM, pipelines de treino (removidos — pertencem ao outro projeto).
- Remote state / backend S3 (state local basta para demo).
- Mudar código Python da aba Áudio.

## Decisions

- **Região default `us-east-1`**: casa com a região hardcoded do Transcribe; bucket em outra região quebraria o job.
- **`force_destroy = true`** (default): permite `destroy` limpar o bucket mesmo com objetos temporários residuais do Transcribe.
- **Nome do bucket via variável** (`transcribe_bucket_name`), default o valor pedido `-psobral89-bucket-transcribe`.
- **`start.sh` lê o nome via `terraform output -raw`** em vez de repetir a variável — fonte única de verdade.
- **`nohup ... &` + PID em arquivo** (`streamlit.pid`): permite ao `stop.sh` matar deterministicamente, com fallback `pgrep -f "streamlit run app.py"`.
- **`stop.sh` mata antes do destroy**: evita objetos sendo escritos durante o destroy.

## Risks / Trade-offs

- **Nome com hífen inicial**: `-psobral89-bucket-transcribe` viola a regra de nomenclatura de bucket S3 (deve começar com letra/número minúsculo) — `apply` deve falhar com `InvalidBucketName`. Mantido como default por ser o valor pedido; ajustar a variável para `psobral89-bucket-transcribe` resolve.
- **State local**: se `start.sh`/`stop.sh` rodarem de máquinas diferentes o state diverge; aceitável para demo individual.
- **`kill` simples (SIGTERM)**: se o Streamlit ignorar, processo pode sobreviver; aceitável, `pgrep` fallback cobre o caso comum.
