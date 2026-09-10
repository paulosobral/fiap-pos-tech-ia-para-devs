#!/usr/bin/env python3
"""
Script para transcrição de vídeo usando faster-whisper.
Uso: python transcribe.py /caminho/do/video.mp4
"""
import subprocess
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def extract_audio(video_path: str, output_audio: str) -> str:
    """Extrai áudio do vídeo."""
    print(f"Extraindo áudio de {video_path}...")
    cmd = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-y", output_audio]
    subprocess.run(cmd, check=True, capture_output=True)
    print("✅ Áudio extraído")
    return output_audio


def transcribe(audio_path: str) -> dict:
    """Transcreve áudio."""
    print("Carregando modelo Whisper...")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    
    print("Transcrevendo...")
    segments, info = model.transcribe(audio_path, language="pt", beam_size=5, vad_filter=True)
    
    full_text = ""
    segments_data = []
    
    for segment in segments:
        full_text += segment.text + " "
        segments_data.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text
        })
    
    return {
        "text": full_text.strip(),
        "segments": segments_data,
        "language": info.language,
        "language_probability": info.language_probability
    }


def main():
    if len(sys.argv) < 2:
        print("Uso: python transcribe.py /caminho/do/video.mp4")
        return 1
    
    video_path = sys.argv[1]
    output_dir = Path(__file__).parent
    audio_output = output_dir / "temp_audio.wav"
    transcription_output = output_dir / f"{Path(video_path).stem}_transcricao.md"
    
    try:
        audio_path = extract_audio(video_path, str(audio_output))
        result = transcribe(str(audio_path))
        
        with open(transcription_output, "w", encoding="utf-8") as f:
            f.write(f"# Transcrição: {Path(video_path).name}\n\n")
            f.write(f"**Idioma:** {result['language']} (confiança: {result['language_probability']:.2%})\n\n")
            f.write("## Transcrição Completa\n\n")
            f.write(result['text'])
            f.write("\n\n")
            f.write("## Transcrição com Timestamps\n\n")
            
            for segment in result['segments']:
                timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
                f.write(f"{timestamp} {segment['text']}\n")
        
        print(f"✅ Transcrição salva: {transcription_output}")
        print(f"📊 Caracteres: {len(result['text'])}")
        
        Path(audio_path).unlink()
        print("🗑️  Arquivo temporário removido")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
