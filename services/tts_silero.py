"""
Генератор речи на основе XTTS с синхронизацией мимики VTube Studio.
Теперь используется потоковая генерация с минимальной задержкой,
человеко-подобной мимикой и гарантированным закрытием рта.
Мимика включается только при старте воспроизведения.
"""

from __future__ import annotations

import os
import time
import threading
import warnings
from typing import Any

import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore

from TTS.tts.models.xtts import Xtts  # type: ignore
from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore

from vtube_controller import VTubeStudioClient

warnings.filterwarnings("ignore", category=UserWarning, module="whisper")
warnings.filterwarnings("ignore", category=UserWarning, module="TTS")

MODEL_PATH = "E:/ElaineRus/models/XTTS-v2"
CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")
SPEAKER_WAV = os.path.join(MODEL_PATH, "samples", "elaine_voice.wav")

config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.config.use_multi_speaker = False  # отключаем поддержку мультирежима для ускорения
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(MODEL_PATH, "model.pth"),
    checkpoint_dir=MODEL_PATH,
    use_deepspeed=False,
)

GPT_COND_LATENT = None
SPEAKER_EMBEDDING = None
try:
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_WAV])
    if torch.cuda.is_available():
        gpt_cond_latent = gpt_cond_latent.cpu().float().cuda()
        speaker_embedding = speaker_embedding.cpu().float().cuda()
        model = model.float().cuda()
    else:
        gpt_cond_latent = gpt_cond_latent.float()
        speaker_embedding = speaker_embedding.float()
        model = model.float().cpu()
    model.eval()
    GPT_COND_LATENT = gpt_cond_latent
    SPEAKER_EMBEDDING = speaker_embedding
except Exception as e:
    print(f"⚠️ Не удалось вычислить латенты голоса: {e}")

vts_client = VTubeStudioClient()

def speak_text(text: str) -> str:
    print("🎣️ Генерация речи через XTTS…")

    if GPT_COND_LATENT is None or SPEAKER_EMBEDDING is None:
        print("⚠️ Латенты голоса не готовы.")
        return ""

    try:
        stream = model.inference_stream(
            text=text,
            language="ru",
            gpt_cond_latent=GPT_COND_LATENT,
            speaker_embedding=SPEAKER_EMBEDDING,
            speed=0.97,
            temperature=1,
            top_k=50,
            top_p=1,
            enable_text_splitting=False,
        )
    except Exception as e:
        print(f"❌ Ошибка stream-генерации: {e}")
        return ""

    sd.default.samplerate = 24000
    sd.default.channels = 1
    out_path = "output/xtts_streamed.wav"
    os.makedirs("output", exist_ok=True)
    audio_accum = []
    stream_finished_flag = threading.Event()
    current_chunk = np.zeros(1, dtype=np.float32)

    def mimics_loop():
        last_volume = 0.0
        smoothing = 0.7
        smoothed_volume = 0.0
        while not stream_finished_flag.is_set():
            rms = np.sqrt(np.mean(np.square(current_chunk)))
            raw_volume = float(np.clip(rms * 4.2, 0.0, 1.0))
            smoothed_volume = smoothing * smoothed_volume + (1 - smoothing) * raw_volume
            if abs(smoothed_volume - last_volume) > 0.01:
                try:
                    vts_client.set_mouth_open(smoothed_volume)
                    last_volume = smoothed_volume
                except Exception:
                    pass
            time.sleep(1 / 60.0)
        try:
            vts_client.set_mouth_open(0.0)
        except Exception:
            pass

    try:
        vts_client.authenticate()
    except Exception:
        pass

    try:
        for chunk in stream:
            chunk_np = chunk.detach().cpu().numpy().astype(np.float32) if isinstance(chunk, torch.Tensor) else np.array(chunk, dtype=np.float32)
            if chunk_np.size == 0:
                continue
            audio_accum.append(chunk_np)
    except Exception as e:
        print(f"⚠️ Ошибка во время генерации: {e}")

    if not audio_accum:
        print("⚠️ Нет сгенерированных аудиоданных.")
        return ""

    full_audio = np.concatenate(audio_accum)
    sf.write(out_path, full_audio, 24000)

    mimics_thread = threading.Thread(target=mimics_loop, daemon=True)
    mimics_thread.start()
    try:
        current_chunk[:] = 0.0
        sd.play(full_audio, samplerate=24000)
        for i in range(0, len(full_audio), 512):
            current_chunk = full_audio[i:i+512]
            time.sleep(512 / 24000)
        sd.wait()
    except Exception as e:
        print(f"⚠️ Ошибка воспроизведения: {e}")
    finally:
        stream_finished_flag.set()
        current_chunk = np.zeros(1, dtype=np.float32)
        try:
            vts_client.set_mouth_open(0.0)
        except Exception:
            pass
        mimics_thread.join(timeout=1.0)

    return out_path
