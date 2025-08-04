"""
Модуль для захвата изображения экрана и генерации комментария с помощью LLM.

Этот файл располагается внутри пакета `services` и использует
относительные импорты для доступа к другим модулям этого пакета.

Для работы требует установленных библиотек `mss`, `Pillow` и `pytesseract`.
В цикле с заданным интервалом делает снимок экрана, передаёт его в функцию
описания и затем получает комментарий от модели. Озвучивает комментарий
с помощью существующего модуля `tts_silero`. Также реализована простая
персистентная память: каждое описание экрана и ответ сохраняются в файл
`memory.log`, чтобы асинхронные процессы могли анализировать предыдущий
контекст при генерации ответов.
"""

from __future__ import annotations

import time
from typing import Optional
import os

# Модули для захвата экрана и обработки изображений. Если mss
# недоступен, мы продолжаем работу, позволяя вызову describe_screen,
# но run_screen_observer будет недоступен.
try:
    import mss  # type: ignore[import]
except ImportError:
    mss = None  # type: ignore
try:
    from PIL import Image  # type: ignore[import]
except ImportError:
    # Pillow необходим для работы описания экрана
    raise RuntimeError(
        "Для модуля screen_capture требуется установить библиотеку Pillow (pip install pillow)"
    )

# OCR библиотека для извлечения текста с экрана
try:
    import pytesseract  # type: ignore[import]
except ImportError:
    pytesseract = None  # type: ignore

# Библиотеки для генерации описания изображения (BLIP). Используем
# отложенную загрузку, чтобы не тратить время и память, если модель
# не требуется. При отсутствии transformers модель не будет использована.
caption_model = None  # type: ignore
caption_processor = None  # type: ignore
caption_available = True

def _load_caption_model() -> bool:
    """
    Загружает модель и процессор для генерации описания изображения
    (BLIP). Возвращает True, если загрузка успешна, иначе False. В случае
    ошибки переводит caption_available в False, чтобы избегать повторных
    попыток загрузки.
    """
    global caption_model, caption_processor, caption_available
    if not caption_available:
        return False
    if caption_model is not None and caption_processor is not None:
        return True
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        # Используем базовую версию модели для экономии ресурсов
        model_name = "Salesforce/blip-image-captioning-base"
        caption_processor = BlipProcessor.from_pretrained(model_name)
        caption_model = BlipForConditionalGeneration.from_pretrained(model_name)
        # Переносим модель на GPU, если доступна CUDA
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                caption_model.to("cuda")
        except Exception:
            pass
        return True
    except Exception:
        # Не удалось загрузить модель — отключаем captioning
        caption_available = False
        return False

def generate_caption(image: Image.Image) -> str:
    """
    Генерирует подпись к изображению с помощью BLIP, если модель доступна.
    Если модель не загружена или произошла ошибка, возвращает пустую строку.

    :param image: PIL.Image
    :return: подпись к изображению
    """
    if not _load_caption_model():
        return ""
    global caption_model, caption_processor
    try:
        import torch  # type: ignore
        # Преобразуем изображение в формат, понятный процессору
        inputs = caption_processor(images=image, return_tensors="pt").to(caption_model.device)
        with torch.no_grad():
            out = caption_model.generate(**inputs, max_new_tokens=64)
        caption = caption_processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception:
        return ""

# Импорт генератора ответа и TTS. Если модуль screen_capture входит в пакет
# services, используем относительные импорты. В противном случае пробуем
# абсолютные импорты из текущей директории.
try:
    from .llm import generate_response, USER_NAME  # type: ignore
    from .tts_silero import speak_text  # type: ignore
except ImportError:
    from llm import generate_response, USER_NAME  # type: ignore
    from tts_silero import speak_text  # type: ignore

MEMORY_FILE = "memory.log"


def _load_memory() -> list[str]:
    """
    Загружает накопленную память из файла. Если файл отсутствует, возвращает пустой список.
    """
    if not os.path.exists(MEMORY_FILE):
        return []
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception:
        return []


def _save_memory(entry: str) -> None:
    """
    Сохраняет новую запись в файл памяти. Каждая запись на новой строке.
    """
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
    except Exception:
        # игнорируем ошибки записи, чтобы не мешать основному процессу
        pass


def extract_text_from_image(image: Image.Image) -> str:
    """
    Извлекает текст из изображения с помощью pytesseract, если библиотека доступна.

    :param image: снимок экрана (PIL.Image)
    :return: строка с распознанным текстом или пустая строка
    """
    if pytesseract is None:
        return ""
    try:
        # Преобразуем изображение в оттенки серого для лучшего качества распознавания
        gray = image.convert("L")
        text = pytesseract.image_to_string(gray, lang="rus+eng")
        return text.strip()
    except Exception:
        return ""


def describe_screen(image: Image.Image) -> str:
    """
    Возвращает краткое описание содержимого скриншота.

    Функция анализирует среднюю яркость, преобладающий цвет изображения и
    извлекает видимый текст (если доступна библиотека OCR). Формирует
    простое описание, например "тёмный экран", "светлый экран с текстом" и
    добавляет первые 200 символов распознанного текста, чтобы модель могла
    понять контекст кода или сайта.

    :param image: снимок экрана (PIL.Image)
    :return: краткое описание
    """
    # Приводим изображение к RGB и вычисляем среднюю яркость каналов
    try:
        from PIL import ImageStat  # late import, Pillow уже загружен
    except Exception:
        # Если Pillow не содержит ImageStat, возвращаем заглушку
        return "неизвестное содержимое экрана"
    # Уменьшаем изображение для ускорения статистики
    thumb = image.copy()
    # Сохраняем пропорции, ограничиваем до 320 px по длинной стороне
    max_dim = max(thumb.width, thumb.height)
    if max_dim > 320:
        scale = 320 / max_dim
        new_size = (int(thumb.width * scale), int(thumb.height * scale))
        thumb = thumb.resize(new_size)
    # Получаем статистику яркости
    stat = ImageStat.Stat(thumb)
    means = stat.mean  # среднее значение по каждому каналу R,G,B
    brightness = sum(means) / 3.0
    # Определяем преобладающий цвет
    dominant_channel = max(enumerate(means), key=lambda x: x[1])[0]
    color_map = {0: "красного", 1: "зелёного", 2: "синего"}
    color_desc = color_map.get(dominant_channel, "неизвестного")
    # Формируем описание по яркости
    if brightness < 50:
        bright_desc = "очень тёмный экран"
    elif brightness < 100:
        bright_desc = "тёмный экран"
    elif brightness < 180:
        bright_desc = "светлый экран"
    else:
        bright_desc = "очень светлый экран"
    summary = f"{bright_desc} с преобладанием {color_desc} цвета"
    # Дополняем описанием текста, если он присутствует
    extracted = extract_text_from_image(image)
    if extracted:
        # Укорачиваем длинный текст, чтобы не перегружать промпт
        short_text = extracted.replace("\n", " ")
        short_text = short_text[:200] + ("…" if len(short_text) > 200 else "")
        summary += f". Текст на экране: {short_text}"
        return summary
    # Если текста на экране нет, пробуем сгенерировать общую подпись
    caption = generate_caption(image)
    if caption:
        return caption
    return summary


def run_screen_observer(interval: float = 10.0, history: Optional[list[str]] = None) -> None:
    """
    Запускает бесконечный цикл наблюдения за экраном. С периодичностью
    `interval` секунд делает снимок экрана, описывает его и
    генерирует голосовой комментарий через LLM и TTS.

    :param interval: интервал между снимками в секундах
    :param history: дополнительная история для контекста (можно
        передавать историю диалога)
    :return: None
    """
    if history is None:
        history = _load_memory()
    if mss is None:
        # Если mss не установлен, информируем пользователя и выходим
        print("⚠️ Библиотека mss не установлена, наблюдение за экраном недоступно.")
        return
    # Используем mss для захвата экрана
    with mss.mss() as sct:
        # Берём первый монитор целиком
        monitor = sct.monitors[1]
        while True:
            # Захват скриншота
            screenshot = sct.grab(monitor)
            # Преобразуем в PIL.Image
            img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            # Получаем описание
            summary = describe_screen(img)
            print(f" Захват экрана: {summary}")
            # Формируем подсказку для модели. Обращаемся к пользователю по имени
            prompt = (
                f"{USER_NAME}, сейчас на экране: {summary}. "
                "Опиши, что ты видишь, и добавь своё краткое мнение или советы."
            )
            # Получаем ответ от модели
            response = generate_response(prompt, history)
            # Озвучиваем ответ
            speak_text(response)
            # Обновляем историю (храним не более 4 записей)
            entry = f"Экран: {summary}\nЭлейн-Сама: {response}"
            history.append(entry)
            if len(history) > 4:
                history[:] = history[-4:]
            # Сохраняем запись в долговременную память
            _save_memory(entry)
            # Ждём до следующего скриншота
            time.sleep(interval)


if __name__ == "__main__":
    """
    Позволяет запускать наблюдение за экраном как отдельный скрипт.

    При запуске файла screen_capture.py напрямую будет выполняться
    функция run_screen_observer() с интервалом по умолчанию. Это
    удобно для автономного наблюдения за экраном, когда голосовая
    активация не требуется.
    """
    try:
        run_screen_observer()
    except KeyboardInterrupt:
        pass