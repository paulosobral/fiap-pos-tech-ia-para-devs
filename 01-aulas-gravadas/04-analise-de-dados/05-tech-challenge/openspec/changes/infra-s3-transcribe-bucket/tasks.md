## 1. Reduzir infra/ ao bucket S3

- [x] 1.1 Adicionar bloco `provider "aws"` em `infra/versions.tf`
- [x] 1.2 Reescrever `infra/variables.tf`: `aws_region` (default `us-east-1`), `transcribe_bucket_name` (default `-psobral89-bucket-transcribe`), `s3_force_destroy`
- [x] 1.3 Reescrever `infra/s3.tf`: apenas bucket + encryption AES256 + public-access-block; remover objetos/prefixos/SageMaker herdados
- [x] 1.4 Reescrever `infra/outputs.tf`: `transcribe_bucket_name` e `transcribe_bucket_arn`
- [x] 1.5 Rodar `terraform fmt` + `terraform validate` (Success)

## 2. Scripts de ciclo de vida na raiz

- [x] 2.1 Criar `start.sh`: `terraform apply -auto-approve`, `source venv/bin/activate`, `nohup streamlit run app.py` com `AUDIO_TRANSCRIBE_BUCKET`, imprime PID (em `streamlit.pid`) e URL
- [x] 2.2 Criar `stop.sh`: mata PID salvo (fallback `pgrep -f "streamlit run app.py"`) e `terraform destroy -auto-approve`
- [x] 2.3 `chmod +x` e `bash -n` nos dois scripts (syntax OK)

## 3. Validação manual (usuário)

- [ ] 3.1 Rodar `./start.sh` com credenciais AWS e confirmar app no browser + bucket criado
- [ ] 3.2 Rodar `./stop.sh` e confirmar processo morto + bucket destruído
- [ ] 3.3 Se `InvalidBucketName`, ajustar `transcribe_bucket_name` para `psobral89-bucket-transcribe` (sem hífen inicial)
