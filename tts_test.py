"""
Генератор речи на основе XTTS с синхронизацией мимики VTube Studio.

Эта реализация работает полностью синхронно: движение рта обновляется
в том же потоке, что и воспроизведение аудио. VTube Studio подключается
один раз при первом вызове и поддерживает соединение с помощью
функционала из `vtube_controller.py`.

Функция `speak_text()` блокирует выполнение до окончания воспроизведения
аудио, что упрощает логику использования. Если требуется параллельная
генерация речи, можно обернуть вызов в отдельный поток.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore

from TTS.tts.models.xtts import Xtts  # type: ignore
from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore

from vtube_controller import VTubeStudioClient

# Пути к модели. Настройте в соответствии с расположением ваших файлов.
MODEL_PATH = "E:/ElaineRus/models/XTTS-v2"
CONFIG_PATH = os.path.join(MODEL_PATH, "config.json")
SPEAKER_WAV = os.path.join(MODEL_PATH, "samples", "elaine_voice.wav")

# Загрузка модели XTTS
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_path=os.path.join(MODEL_PATH, "model.pth"),
    checkpoint_dir=MODEL_PATH,
)
if torch.cuda.is_available():
    model.cuda()
else:
    model.cpu()
model.eval()

# Инициализация клиента VTube Studio (с постоянным соединением)
vts_client = VTubeStudioClient()


def speak_text(text: str) -> str:
    """
    Генерирует аудиофайл из текста, проигрывает его и синхронно обновляет
    параметр `MouthOpen` в VTube Studio. Возвращает путь к сохранённому WAV.
    """
    print("🗣️ Генерация речи через XTTS…")
    # Синтезируем аудио
    output: dict[str, Any] = model.synthesize(
        text=text,
        config=config,
        speaker_wav=SPEAKER_WAV,
        language="ru",
    )
    audio: np.ndarray = output["wav"]

    # Обеспечиваем аутентификацию (только при первом вызове)
    try:
        vts_client.authenticate()
    except Exception as e:
        print(f"⚠️ Ошибка аутентификации VTube Studio: {e}")

    # Сохраняем WAV
    os.makedirs("output", exist_ok=True)
    out_path = "output/xtts_output.wav"
    sf.write(out_path, audio, 24000)

    # Запускаем воспроизведение
    sd.play(audio, samplerate=24000)
    frame_duration = 1 / 30.0
    samples_per_frame = int(24000 * frame_duration)

    # Обновляем мимику каждые ~33 мс
    for start in range(0, len(audio), samples_per_frame):
        end = start + samples_per_frame
        chunk = audio[start:end]
        if chunk.size == 0:
            break
        volume = float(np.clip(np.abs(chunk).mean() * 5.0, 0.0, 1.0))
        try:
            vts_client.set_mouth_open(volume)
        except Exception as e:
            print(f"⚠️ Ошибка отправки MouthOpen: {e}")
        time.sleep(frame_duration)
    # Закрываем рот и ждём завершения аудио
    try:
        vts_client.set_mouth_open(0.0)
    except Exception as e:
        print(f"⚠️ Ошибка при закрытии рта: {e}")
    sd.wait()
    sd.stop()
    return out_path