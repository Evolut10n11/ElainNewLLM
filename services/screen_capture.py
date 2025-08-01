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

try:
    import mss  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:
    # Если mss или Pillow не установлены, сообщаем об этом при импорте
    raise RuntimeError(
        "Для модуля screen_capture требуется установить библиотеки mss и Pillow (pip install mss pillow)"
    )

from .llm import generate_response, USER_NAME
from .tts_silero import speak_text


def describe_screen(image: Image.Image) -> str:
    """
    Возвращает текстовое описание изображённого на экране.

    На первом этапе это заглушка, возвращающая фиксированное
    сообщение. Позднее сюда можно встроить OCR или captioning.

    :param image: снимок экрана (PIL.Image)
    :return: краткое описание
    """
    # TODO: заменить на реальное описание картинки с помощью OCR или модели captioning
    return "неизвестное содержимое экрана"


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