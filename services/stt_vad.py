import subprocess
import tempfile
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time

# ПУТИ К WHISPER
WHISPER_BIN = "E:/ElaineRus/whisper.cpp/build/bin/Release/whisper.exe"
WHISPER_MODEL = "E:/ElaineRus/models/whisper/ggml-medium.bin"

# VAD-ПАРАМЕТРЫ
THRESHOLD = 500
SAMPLE_RATE = 16000
MIN_DURATION = 0.5
MAX_RECORD_SECONDS = 3.5


def wait_for_voice(threshold=THRESHOLD, timeout=10):
    """Ожидает появления речи на микрофоне."""
    print("🎙 Жду начала речи...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        audio = sd.rec(int(SAMPLE_RATE * 0.5), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        volume = max(abs(audio.flatten()))
        if volume > threshold:
            print("🟢 Обнаружен голос, начинаю запись...")
            return True
    print("🔇 Голос не обнаружен — таймаут.")
    return False


def record_vad(seconds=MAX_RECORD_SECONDS) -> str:
    """Записывает звук с микрофона после VAD и сохраняет во временный WAV."""
    if not wait_for_voice():
        return ""
    audio = sd.rec(int(SAMPLE_RATE * seconds), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    volume = max(abs(audio.flatten()))
    if volume < THRESHOLD:
        print("🔇 Слишком тихо, не записываю.")
        return ""
    os.makedirs("output", exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="output") as f:
        sf.write(f.name, audio, SAMPLE_RATE, subtype='PCM_16')
        return f.name


def clean_transcript(text: str) -> str:
    """Удаляет вставки и шум из текста."""
    trash_phrases = [
        "редактор субтитров", "музыка", "[", "]", "♪", "applause", "noise",
        "заставка", "смех", "кашель", "речь", "переход", "звуки"
    ]
    for phrase in trash_phrases:
        text = text.replace(phrase, "")
    return text.strip()


def transcribe_vad(wav_path: str) -> str:
    """Распознаёт WAV-файл с помощью whisper.cpp."""
    print("🧠 Распознаём голос через whisper.exe...")

    info = sf.info(wav_path)
    if info.duration < MIN_DURATION:
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
