import torch
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import asyncio

from vtube_controller import vts_client
from services.stt_vad import LAST_SPEAK_END

# Путь до модели Silero TTS
MODEL_PATH = "E:/ElaineRus/models/silero/v3_1_ru.pt"
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
model.to(device)
sample_rate = 24000

def speak_text(text: str) -> float:
    global LAST_SPEAK_END

    if not text.strip():
        print("⚠️ Пустой текст для озвучки.")
        return 0.0

    start = time.time()
    audio = model.apply_tts(
        text,
        speaker="baya",
        sample_rate=sample_rate,
        put_accent=True,
        put_yo=True
    )
    audio = audio / np.abs(audio).max()

    os.makedirs("output", exist_ok=True)
    out_path = f"output/silero_{int(start)}.wav"
    sf.write(out_path, audio, sample_rate)
    sd.play(audio, sample_rate)

    duration = len(audio) / sample_rate

    async def lipsync(dur: float):
        await vts_client.set_mouth_open(1.0)
        await asyncio.sleep(dur)
        await vts_client.set_mouth_open(0.0)

    asyncio.run(lipsync(duration))
    sd.wait()

    # Помечаем время конца воспроизведения
    LAST_SPEAK_END = time.time()

    print(f"TTS готово за {time.time() - start:.2f} сек")
    return duration
