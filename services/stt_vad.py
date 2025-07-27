import subprocess
import tempfile
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time

WHISPER_BIN = "E:/ElaineRus/whisper.cpp/build/bin/Release/whisper.exe"
WHISPER_MODEL = "E:/ElaineRus/models/whisper/ggml-medium.bin"
FFMPEG_PATH = "E:/ffmpeg-7.1.1-essentials_build/bin/ffmpeg.exe"


def wait_for_voice(threshold=500, timeout=10, samplerate=16000):
    print("🎙 Жду начала речи...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        audio = sd.rec(int(samplerate * 0.5), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        volume = max(abs(audio.flatten()))
        if volume > threshold:
            print("🟢 Обнаружен голос, начинаю запись...")
            return True
    print("🔇 Голос не обнаружен — таймаут.")
    return False


def record_vad(seconds=2.5, samplerate=16000) -> str:
    if not wait_for_voice():
        return ""

    audio = sd.rec(int(samplerate * seconds), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()

    volume = max(abs(audio.flatten()))
    if volume < 500:
        print("🔇 Слишком тихо, не записываю.")
        return ""

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate, subtype='PCM_16')
        return f.name

def clean_transcript(text: str) -> str:
    trash_phrases = ["редактор субтитров", "[", "]", "музыка", "applause", "noise", "♪"]
    for phrase in trash_phrases:
        text = text.replace(phrase, "")
    return text.strip()

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

    print("🧠 Распознаём голос через whisper.exe...")
    try:
        out = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            timeout=15
        )
        lines = out.splitlines()
        text = "\n".join([l for l in lines if "-->" in l])
        result = text.split("]")[-1].strip()
        return clean_transcript(result)
    except subprocess.TimeoutExpired:
        print("⏳ Whisper завис — превышен лимит времени.")
        return ""
    except subprocess.CalledProcessError as e:
        print("❌ Whisper ошибка:\n", e.output)
        return ""
    finally:
        os.remove(wav_path)
        if os.path.exists(fixed_path):
            os.remove(fixed_path)