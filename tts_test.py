# services/tts_xtts.py
from TTS.tts.models.xtts import Xtts
from TTS.tts.configs.xtts_config import XttsConfig
import torch
import soundfile as sf
import os

MODEL_PATH = "E:/ElaineRus/models/XTTS-v2"
CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")
SPEAKER_WAV = os.path.join(MODEL_PATH, "samples", "elaine_voice.wav")  # Убедись, что файл существует

# Загружаем конфиг и модель
config = XttsConfig()
config.load_json(CONFIG_PATH)

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=os.path.join(MODEL_PATH, "model.pth"))
model.cuda() if torch.cuda.is_available() else model.cpu()
model.eval()

def speak_text(text: str) -> str:
    output = model.synthesize(
        text=text,
        config=config,
        speaker_wav=SPEAKER_WAV,
        language="ru"
    )
    audio = output["wav"]
    out_path = "output/xtts_output.wav"
    sf.write(out_path, audio, 24000)
    return out_path
