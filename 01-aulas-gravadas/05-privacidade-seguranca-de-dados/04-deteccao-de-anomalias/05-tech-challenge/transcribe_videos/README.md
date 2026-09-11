# Transcrição de Vídeo com faster-whisper

Script para transcrição de vídeos em Português Brasileiro usando faster-whisper (versão leve do Whisper da OpenAI).

## 🚀 Uso

```bash
python transcribe.py /caminho/do/video.mp4
```

**Exemplo:**
```bash
python transcribe.py "/home/usuario/videos/aula.mp4"
```

## 📋 Pré-requisitos

### Dependências do Sistema

1. **ffmpeg** (para extração de áudio de vídeo):
   ```bash
   sudo dnf install ffmpeg
   ```

### Dependências Python

```bash
pip install faster-whisper
```

## 📝 Como Funciona

O script:
1. Extrai o áudio do vídeo usando ffmpeg
2. Baixa o modelo Whisper automaticamente (primeira execução, ~150MB)
3. Transcreve usando faster-whisper (offline, PT-BR)
4. Salva a transcrição em formato Markdown
5. Remove arquivos temporários

## 📁 Arquivo de Saída

Gera arquivo `[nome_do_video]_transcricao.md` com:
- Transcrição completa do texto
- Transcrição com timestamps por segmento
- Idioma detectado e confiança

**Exemplo:** Se você rodar `python transcribe.py aula.mp4`, vai gerar `aula_transcricao.md`

## 📊 Resultado do Teste

Testado com sucesso no vídeo "21 - Mentoria.mp4":
- **52.140 caracteres** transcritos
- **Idioma detectado:** Português (100% de confiança)
- **Tempo de processamento:** ~4 minutos para vídeo de 2 horas
- **Qualidade:** Excelente, com contexto e precisão

## 🎯 Características

- ✅ Funciona offline (após download do modelo)
- ✅ Suporte nativo a Português Brasileiro
- ✅ Timestamps precisos
- ✅ Alta qualidade de transcrição
- ✅ Gratuito
