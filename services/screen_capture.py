"""
Модуль для захвата изображения экрана и генерации комментария с помощью LLM.

Этот файл располагается внутри пакета `services` и использует
относительные импорты для доступа к другим модулям этого пакета.

Для работы требует установленных библиотек `mss` и `Pillow`. В цикле
с заданным интервалом делает снимок экрана, передаёт его в функцию
описания и затем получает комментарий от модели. Озвучивает
комментарий с помощью существующего модуля `tts_silero`.

Функция `describe_screen` пока является заглушкой. Её следует
заменить на реальное распознавание (OCR, captioning и т. п.), когда
модели будут готовы. На данный момент она возвращает статичное
описание, чтобы проверить интеграцию.
"""

from __future__ import annotations

import time
from typing import Optional

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

# Импорт генератора ответа и TTS. Если модуль screen_capture входит в пакет
# services, используем относительные импорты. В противном случае пробуем
# абсолютные импорты из текущей директории.
try:
    from .llm import generate_response, USER_NAME  # type: ignore
    from .tts_silero import speak_text  # type: ignore
except ImportError:
    from llm import generate_response, USER_NAME  # type: ignore
    from tts_silero import speak_text  # type: ignore


def describe_screen(image: Image.Image) -> str:
    """
    Возвращает краткое описание содержимого скриншота.

    Функция анализирует среднюю яркость и преобладающий цвет
    изображения и формирует простое описание, например
    "тёмный экран" или "светлый экран с преобладанием синего".
    Если в дальнейшем будет добавлено OCR или captioning,
    реализацию можно расширить.

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
    # Добавляем информацию о цветовом доминировании
    return f"{bright_desc} с преобладанием {color_desc} цвета"


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
        history = []
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
            print(f"📸 Захват экрана: {summary}")
            # Формируем подсказку для модели. Обращаемся к пользователю по имени
            prompt = (
                f"{USER_NAME}, сейчас на экране: {summary}. "
                "Опиши, что ты видишь, и добавь своё краткое мнение."
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
            # Ждём до следующего скриншота
            time.sleep(interval)


if __name__ == "__main__":
    """
    Позволяет запускать наблюдение за экраном как отдельный скрипт.

    При запуске файла screen_capture.py напрямую будет
    выполняться функция run_screen_observer() с интервалом по
    умолчанию. Это удобно для автономного наблюдения за
    экраном, когда голосовая активация не требуется.
    """
    try:
        run_screen_observer()
    except KeyboardInterrupt:
        print("🛑 Остановка наблюдения за экраном по Ctrl+C")