## ADDED Requirements

### Requirement: Provisionamento do bucket S3 do Transcribe

O Terraform em `infra/` SHALL provisionar exatamente um bucket S3 destinado ao AWS Transcribe da aba Áudio, com nome configurável via variável e sem depender de nenhum recurso alheio a este projeto (SageMaker, IAM roles, objetos de treino).

#### Scenario: Apply cria o bucket

- **WHEN** `terraform apply -auto-approve` roda em `infra/` com credenciais AWS válidas
- **THEN** um bucket S3 com o nome de `var.transcribe_bucket_name` é criado na região `var.aws_region`
- **AND** nenhum recurso SageMaker/IAM é criado

#### Scenario: Configuração válida e autossuficiente

- **WHEN** `terraform validate` roda em `infra/`
- **THEN** retorna sucesso sem referências a recursos indefinidos
- **AND** um bloco `provider "aws"` está presente

### Requirement: Segurança do bucket

O bucket SHALL ter criptografia server-side padrão (AES256) e bloqueio total de acesso público.

#### Scenario: Encryption e public-access-block aplicados

- **WHEN** o bucket é criado
- **THEN** a criptografia padrão é AES256
- **AND** todas as quatro flags de `aws_s3_bucket_public_access_block` estão `true`

### Requirement: Outputs do bucket

O Terraform SHALL expor o nome e o ARN do bucket como outputs para consumo pelos scripts de ciclo de vida.

#### Scenario: Output do nome disponível

- **WHEN** `terraform output -raw transcribe_bucket_name` é executado após o apply
- **THEN** retorna o nome do bucket criado

### Requirement: Script de subida (start.sh)

Um `start.sh` na raiz do projeto SHALL, em sequência: rodar `terraform apply -auto-approve` em `infra/`, ativar o venv, e subir o Streamlit via `nohup` com a env `AUDIO_TRANSCRIBE_BUCKET` apontando para o bucket criado, imprimindo o PID e a URL de conexão.

#### Scenario: Subida completa

- **WHEN** `./start.sh` é executado
- **THEN** o bucket é provisionado via apply
- **AND** o Streamlit inicia em background com `AUDIO_TRANSCRIBE_BUCKET` igual ao output do bucket
- **AND** o PID é salvo em arquivo e impresso junto com a URL `http://localhost:<porta>`

### Requirement: Script de derrubada (stop.sh)

Um `stop.sh` na raiz do projeto SHALL localizar e matar o processo do Streamlit e, em seguida, rodar `terraform destroy -auto-approve` em `infra/`.

#### Scenario: Derrubada completa

- **WHEN** `./stop.sh` é executado com o Streamlit rodando
- **THEN** o processo do Streamlit é encerrado (via PID salvo ou fallback por `pgrep`)
- **AND** o bucket S3 é destruído via `terraform destroy -auto-approve`
