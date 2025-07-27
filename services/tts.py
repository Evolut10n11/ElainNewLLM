import os
import time
import sounddevice as sd
import soundfile as sf
from TTS.api import TTS

#  ——————————————————————————————————————————————————————————————
# 1) указываем модель с поддержкой cloning
#    этот репозиторий уже умеет принимать speaker_wav
TTS_MODEL = "tts_models/multilingual/multi-dataset/your_tts"
tts = TTS(model_name=TTS_MODEL, gpu=False)   # gpu=True, если у вас CUDA

# 2) указываем свой семпл (ваш WAV)
VOICE_SAMPLE = os.path.join(os.getcwd(), "voice_samples", "neuro-sama-tts-file.wav")

# 3) папка для выходных WAV
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def speak(text: str) -> str:
    """
    Клонирует голос из VOICE_SAMPLE и синтезирует text.
    Сохраняет WAV и сразу воспроизводит.
    Возвращает путь к файлу.
    """
    # формируем путь
    ts = int(time.time())
    out_path = os.path.join(OUTPUT_DIR, f"tts_clone_{ts}.wav")

    # синтез в файл (Coqui TTS API)
    tts.tts_to_file(
        text=text,
        speaker_wav=VOICE_SAMPLE,
        file_path=out_path
    )

    # воспроизведение
    data, sr = sf.read(out_path)
    sd.play(data, sr)
    sd.wait()

    return out_path
