import subprocess
import tempfile
import soundfile as sf
import sounddevice as sd
import numpy as np
import os
import time

WHISPER_BIN   = "E:/ElaineRus/whisper.cpp/build/bin/Release/whisper.exe"
WHISPER_MODEL = "E:/ElaineRus/models/whisper/ggml-medium.bin"

# Таймстамп конца последнего TTS-воспроизведения
LAST_SPEAK_END = 0.0

def wait_for_voice(threshold=500, timeout=10, samplerate=16000):
    global LAST_SPEAK_END
    now = time.time()
    # Если только что был TTS — ждём, пока эхо спадёт
    if now - LAST_SPEAK_END < 1.0:
        time.sleep(1.0 - (now - LAST_SPEAK_END))

    print("🎙 Жду начала речи...")
    start = time.time()
    while time.time() - start < timeout:
        audio = sd.rec(int(samplerate*0.5), samplerate=samplerate, channels=1, dtype='int16')
        sd.wait()
        if max(abs(audio.flatten())) > threshold:
            print("🟢 Обнаружен голос, начинаю запись...")
            return True
    print("🔇 Голос не обнаружен — таймаут.")
    return False

def record_vad(seconds=2.5, samplerate=16000) -> str:
    if not wait_for_voice():
        return ""
    audio = sd.rec(int(samplerate*seconds), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    if max(abs(audio.flatten())) < 500:
        print("🔇 Слишком тихо, не записываю.")
        return ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        sf.write(f.name, audio, samplerate, subtype='PCM_16')
        return f.name

def clean_transcript(text: str) -> str:
    trash = ["редактор субтитров","[","]","музыка","applause","noise","♪"]
    for t in trash:
        text = text.replace(t, "")
    return text.strip()

def transcribe_vad(wav_path: str) -> str:
    print("🧠 Распознаём голос через whisper.exe...")
    info = sf.info(wav_path)
    if info.duration < 0.5:
        print("⚠️ Файл слишком короткий, пропускаю.")
        os.remove(wav_path)
        return ""
    cmd = [WHISPER_BIN, "-m", WHISPER_MODEL, "-l", "ru", "--threads", "8", wav_path]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, encoding="utf-8", timeout=15).stdout
        lines = [l for l in out.splitlines() if "-->" in l]
        res = lines[-1].split("]")[-1].strip() if lines else ""
        return clean_transcript(res)
    except subprocess.TimeoutExpired:
        print("⏳ Whisper завис — превышен лимит времени.")
        return ""
    except subprocess.CalledProcessError as e:
        print("❌ Whisper ошибка:", e.output)
        return ""
    finally:
        os.remove(wav_path)
