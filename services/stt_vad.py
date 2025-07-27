import subprocess
import tempfile
import soundfile as sf
import sounddevice as sd
import os

WHISPER_BIN = "E:/ElaineRus/whisper.cpp/build/bin/Release/whisper.exe"
WHISPER_MODEL = "E:/ElaineRus/models/whisper/ggml-medium.bin"
FFMPEG_PATH = "E:/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"

def record_vad(seconds=3, samplerate=16000) -> str:
    print("🎤 Говорите…")
    audio = sd.rec(int(samplerate * seconds), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate, subtype='PCM_16')
        return f.name

def transcribe_vad(wav_path: str) -> str:
    print("🔄 Преобразуем WAV в совместимый формат через ffmpeg...")
    with tempfile.NamedTemporaryFile(suffix="_fixed.wav", delete=False) as fixed:
        fixed_path = fixed.name

    subprocess.run([
        FFMPEG_PATH, "-y",
        "-i", wav_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        fixed_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-l", "ru",
        "-np",
        fixed_path
    ]

    print("🧠 Распознаём голос через whisper.exe VAD...")
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        lines = out.splitlines()
        text = "\n".join([l for l in lines if "-->" in l])
        result = text.split("]")[-1].strip()
        return result
    except subprocess.CalledProcessError as e:
        print("❌ Whisper VAD ошибка:\n", e.output)
        return ""
    finally:
        os.remove(wav_path)
        if os.path.exists(fixed_path):
            os.remove(fixed_path)
