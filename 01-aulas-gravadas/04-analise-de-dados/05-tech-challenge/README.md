# Tech Challenge Fase 4 — Monitoramento Multimodal de Pacientes

> Documentação completa (setup detalhado, arquitetura, uso de cada aba) será escrita na Task 8.
> Esta seção é um stub mínimo criado na Task 1 (setup do projeto).

## Pré-requisitos

- Python 3.11+
- Credenciais AWS já configuradas em `~/.aws/credentials` (perfis `default` e `bedrock`), com permissão para os serviços **Transcribe**, **Comprehend** e **Bedrock** (modelo Claude Sonnet habilitado na região utilizada). Não é necessário reconfigurar — o arquivo já existe neste ambiente.

## Setup rápido

```bash
cd 01-aulas-gravadas/04-analise-de-dados/05-tech-challenge/
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Estrutura de diretórios (uma capability por pasta, ver `openspec/changes/monitoramento-multimodal-pacientes/design.md`, decisão D7):

```
anomaly/        # detecção de anomalia compartilhada (rolling z-score)
video/          # análise de vídeo (YOLOv8-pose)
audio/          # análise de áudio (AWS Transcribe + Comprehend)
vital_signs/    # sinais vitais (z-score + Isolation Forest)
prescriptions/  # revisão de prescrições (AWS Bedrock)
alerts/         # feed de alertas compartilhado
data/           # datasets/mídia de demonstração
```
