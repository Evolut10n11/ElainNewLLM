import torch
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf

MODEL_PATH = "E:/ElaineRus/models/silero/v3_1_ru.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
model.to(device)

sample_rate = 24000

def speak_text(text: str) -> str:
    if not text.strip():
        print("⚠️ Пустой текст для озвучки.")
        return ""

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
    out_path = f"output/silero_{int(time.time())}.wav"
    sf.write(out_path, audio, sample_rate)
    sd.play(audio, sample_rate)
    sd.wait()

    print(f"TTS готов за {time.time() - start:.2f} сек")
    return out_path
