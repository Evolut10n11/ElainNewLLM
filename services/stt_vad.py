import subprocess
import tempfile
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time

WHISPER_BIN = "E:/ElaineRus/whisper.cpp/build/bin/Release/whisper.exe"
WHISPER_MODEL = "E:/ElaineRus/models/whisper/ggml-medium.bin"

def wait_for_voice(threshold=500, timeout=10, samplerate=16000):
    """Ожидает появления голоса, используя порог по громкости."""
    print("🎙 Жду начала речи...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        audio = sd.rec(int(samplerate * 0.5), samplerate=samplerate,
                       channels=1, dtype='int16')
        sd.wait()
        volume = max(abs(audio.flatten()))
        if volume > threshold:
            print("🟢 Обнаружен голос, начинаю запись...")
            return True
    print("🔇 Голос не обнаружен — таймаут.")
    return False

def record_vad(seconds=2.5, samplerate=16000) -> str:
    """Записывает звук заданной длительности после обнаружения голоса. Возвращает путь к WAV-файлу."""
    if not wait_for_voice():
        return ""
    audio = sd.rec(int(samplerate * seconds), samplerate=samplerate,
                   channels=1, dtype='int16')
    sd.wait()
    volume = max(abs(audio.flatten()))
    if volume < 500:
        print("🔇 Слишком тихо, не записываю.")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate, subtype='PCM_16')
        return f.name

def clean_transcript(text: str) -> str:
    """Очищает распознанный текст от нежелательных вставок."""
    trash_phrases = ["редактор субтитров", "[", "]", "музыка", "applause",
                     "noise", "♪"]
    for phrase in trash_phrases:
        text = text.replace(phrase, "")
    return text.strip()

def transcribe_vad(wav_path: str) -> str:
    """Распознаёт речь из WAV-файла с помощью Whisper."""
    print("🧠 Распознаём голос через whisper.exe...")
    info = sf.info(wav_path)
    if info.duration < 0.5:
        print("⚠️ Файл слишком короткий, пропускаю.")
        os.remove(wav_path)
        return ""
    cmd = [
        WHISPER_BIN,
        "-m", WHISPER_MODEL,
        "-l", "ru",
        "--threads", "8",
        wav_path
    ]
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, encoding="utf-8", timeout=15
        ).stdout
        lines = [l for l in out.splitlines() if "-->" in l]
        if not lines:
            return ""
        result = lines[-1].split("]")[-1].strip()
        return clean_transcript(result)
    except subprocess.TimeoutExpired:
        print("⏳ Whisper завис — превышен лимит времени.")
        return ""
    except subprocess.CalledProcessError as e:
        print("❌ Whisper ошибка:", e.output)
        return ""
    finally:
        os.remove(wav_path)
