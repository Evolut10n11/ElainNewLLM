import asyncio
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
import torch
import soundfile as sf
import os
import sounddevice as sd
import numpy as np
from vtube_controller import VTubeStudioClient  # твой файл

MODEL_PATH = "E:/ElaineRus/models/XTTS-v2"
CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")
SPEAKER_WAV = os.path.join(MODEL_PATH, "samples", "elaine_voice.wav")

config = XttsConfig()
config.load_json(CONFIG_PATH)

model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(MODEL_PATH, "model.pth"),
    checkpoint_dir=MODEL_PATH
)
model.cuda() if torch.cuda.is_available() else model.cpu()
model.eval()

vts_client = VTubeStudioClient()

def speak_text(text: str) -> str:
    print("🗣️ Генерация речи через XTTS…")
    output = model.synthesize(
        text=text,
        config=config,
        speaker_wav=SPEAKER_WAV,
        language="ru"
    )
    audio = output["wav"]

    os.makedirs("output", exist_ok=True)
    out_path = "output/xtts_output.wav"
    sf.write(out_path, audio, 24000)

    async def play_with_mouth():
        try:
            await vts_client.authenticate()
            sd.play(audio, samplerate=24000)

            frame_duration = 1 / 30
            samples_per_frame = int(24000 * frame_duration)
            total_frames = len(audio) // samples_per_frame

            for i in range(total_frames):
                start = i * samples_per_frame
                end = start + samples_per_frame
                chunk = audio[start:end]
                volume = float(np.clip(np.abs(chunk).mean() * 5.0, 0.0, 1.0))
                await vts_client.set_mouth_open(volume)
                await asyncio.sleep(frame_duration)

            await vts_client.set_mouth_open(0.0)
        except Exception as e:
            print(f"⚠️ Ошибка движения рта: {e}")
        finally:
            sd.stop()

    asyncio.run(play_with_mouth())
    return out_path
