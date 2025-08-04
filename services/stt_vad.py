import whisper
import numpy as np
import sounddevice as sd
import torch

THRESHOLD = 500
SAMPLE_RATE = 16000
MAX_RECORD_SECONDS = 3.5
MIN_DURATION = 0.5

# Загружаем модель один раз (medium на русском)
whisper_model = whisper.load_model("medium").to("cuda" if torch.cuda.is_available() else "cpu")

def wait_for_voice(threshold=THRESHOLD, timeout=10):
    print("🎙 Жду начала речи...")
    for _ in range(int(timeout * 2)):
        audio = sd.rec(int(SAMPLE_RATE * 0.5), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        volume = max(abs(audio.flatten()))
        if volume > threshold:
            print("🟢 Обнаружен голос, начинаю запись...")
            return True
    print("🔇 Голос не обнаружен — таймаут.")
    return False

def record_vad(seconds=MAX_RECORD_SECONDS) -> np.ndarray:
    if not wait_for_voice():
        return np.array([], dtype=np.int16)
    audio = sd.rec(int(SAMPLE_RATE * seconds), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    volume = max(abs(audio.flatten()))
    if volume < THRESHOLD:
        print("🔇 Слишком тихо, не записываю.")
        return np.array([], dtype=np.int16)
    return audio.flatten()

def transcribe_vad(audio: np.ndarray) -> str:
    print("🧠 Распознаём голос через Whisper (CUDA)…")
    if len(audio) < int(SAMPLE_RATE * MIN_DURATION):
        print("⚠️ Слишком короткий фрагмент.")
        return ""
    
    audio = audio.astype(np.float32) / 32768.0  # int16 → float32
    result = whisper_model.transcribe(audio, language="ru", fp16=torch.cuda.is_available())
    return clean_transcript(result["text"])

def clean_transcript(text: str) -> str:
    trash_phrases = [
        "редактор субтитров", "музыка", "[", "]", "♪", "applause", "noise",
        "заставка", "смех", "кашель", "речь", "переход", "звуки"
    ]
    for phrase in trash_phrases:
        text = text.replace(phrase, "")
    return text.strip()