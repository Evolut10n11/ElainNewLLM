import torch
import time
import os
import numpy as np
import sounddevice as sd
import soundfile as sf
import asyncio
from vtube_controller import VTubeStudioClient

# Создаём отдельный цикл событий для работы с VTube Studio. Это позволяет
# использовать одно и то же соединение WebSocket в рамках одного
# event loop, избегая ошибок «no close frame received or sent».
_event_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_event_loop)

# Путь до модели Silero TTS
MODEL_PATH = "E:/ElaineRus/models/silero/v3_1_ru.pt"
# Используем GPU, если доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Загрузка модели Silero
model = torch.package.PackageImporter(MODEL_PATH).load_pickle("tts_models", "model")
model.to(device)
# Частота дискретизации модели
sample_rate = 24000

# Глобальный клиент VTube Studio (создаём один раз)
_vts_client = VTubeStudioClient()

# Аутентифицируемся в VTube Studio сразу при загрузке модуля. Это обеспечивает
# подключение при старте программы, чтобы не запрашивать разрешение в момент
# первого синтеза речи. Повторный вызов authenticate() в дальнейшем
# пропускается благодаря флагу authenticated в VTubeStudioClient.
try:
    _event_loop.run_until_complete(_vts_client.authenticate())
except Exception as _e:
    # Если аутентификация не удалась (например, VTube Studio не запущена),
    # выводим предупреждение, но продолжаем работу. Попытка подключения
    # произойдёт позже при синтезе речи.
    print(f"⚠️ Не удалось аутентифицироваться в VTube Studio при старте: {_e}")

def speak_text(text: str) -> str:
    """
    Генерирует аудио из текста, воспроизводит его и синхронизирует рот аватара.
    Возвращает путь к сохранённому WAV-файлу с озвучкой.
    """
    # Аутентификация в VTube Studio (при необходимости запрашивает Allow один раз)
    try:
        # Выполняем аутентификацию в рамках общего цикла событий
        _event_loop.run_until_complete(_vts_client.authenticate())
    except Exception as e:
        print(f"🔌 Ошибка при аутентификации VTube Studio: {e}")
        return ""
    if not text.strip():
        print("⚠️ Пустой текст для озвучки.")
        return ""
    # Генерация аудио с помощью модели Silero
    start_time = time.time()
    audio = model.apply_tts(
        text,
        speaker="baya",
        sample_rate=sample_rate,
        put_accent=True,
        put_yo=True
    )
    # Преобразуем выход в numpy-массив. model.apply_tts может возвращать
    # тензор PyTorch, для которого numpy.max вызывает torch.max с
    # неподдерживаемыми аргументами. Прямое преобразование позволяет
    # корректно использовать функции numpy.
    if hasattr(audio, "cpu"):
        audio = audio.detach().cpu().numpy()
    # Нормализация громкости (на всякий случай)
    audio = audio / np.max(np.abs(audio))
    # Сохранение аудио в файл (можно убрать, если файл не нужен)
    os.makedirs("output", exist_ok=True)
    out_path = f"output/silero_{int(start_time)}.wav"
    sf.write(out_path, audio, sample_rate)
    # Воспроизведение аудио (асинхронно)
    sd.play(audio, sample_rate)
    duration = len(audio) / sample_rate

    # Асинхронная функция для циклической отправки значений MouthOpen
    async def _lipsync(audio_array: np.ndarray, dur: float):
        chunk_dur = 0.2  # интервал обновления 0.2 с (5 раз/сек)
        chunk_samples = int(chunk_dur * sample_rate)
        try:
            # Проходимся по аудиоданным с шагом chunk_samples
            for i in range(0, len(audio_array), chunk_samples):
                chunk = audio_array[i:i+chunk_samples]
                # Рассчитываем громкость фрагмента (максимальное отклонение)
                if len(chunk) == 0:
                    value = 0.0
                else:
                    value = float(np.max(np.abs(chunk)))
                # Отправляем значение MouthOpen (0.0-1.0) в VTube Studio
                await _vts_client.set_mouth_open(value)
                # Ждём 0.2 с перед отправкой следующего значения
                await asyncio.sleep(chunk_dur)
            # После окончания фразы закрываем рот (значение 0.0)
            await _vts_client.set_mouth_open(0.0)
        except Exception as e:
            print("🔌 Ошибка синхронизации рта:", e)

    # Запускаем асинхронный цикл синхронизации рта
    # Запускаем цикл синхронизации рта в том же loop
    _event_loop.run_until_complete(_lipsync(audio, duration))
    # Ожидаем окончания воспроизведения аудио
    sd.wait()
    print(f"TTS готово за {time.time() - start_time:.2f} сек")
    return out_path
