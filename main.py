"""
Точка входа в приложение: объединяет распознавание речи,
генерацию ответа и синтез речи. Модули могут находиться либо в
пакете `services`, либо в текущем каталоге. Для переносимости
пробуем импортировать из `services` и, при неудаче, используем
локальные файлы.
"""

# Пытаемся импортировать из пакета services. Если он отсутствует (например,
# в упрощенной структуре проекта), импортируем из текущей директории.
try:
    from services.stt_vad import record_vad, transcribe_vad
    from services.llm import generate_response, USER_NAME
    from services.tts_silero import speak_text
except ImportError:
    from stt_vad import record_vad, transcribe_vad  # type: ignore
    from llm import generate_response, USER_NAME  # type: ignore
    from tts_silero import speak_text  # type: ignore
from time import sleep

def main():
    last_text = None
    last_response = None
    chat_history = []
    while True:
        wav = record_vad()
        if not wav:
            continue
        user_text = transcribe_vad(wav)
        if not user_text.strip():
            print("😶 Ничего не распознано. Попробуйте ещё раз.")
            continue
        print(f"Вы сказали: {user_text}")
        # Проверка на повторяющийся вопрос (во избежание эха)
        if last_text and user_text.strip().lower().startswith(last_text.strip().lower()):
            print("🔁 Похоже на повтор (вопрос) — пропускаю...")
            continue
        # Генерация ответа (история используется при необходимости)
        response = generate_response(user_text, chat_history)
        # Проверка, завершено ли предложение пользователя
        if user_text.strip()[-1] not in ".!?…»":
            print("🟡 Похоже, ты не договорил — не добавляю запрос в историю.")
        # Проверка, завершён ли ответ модели
        if response.strip()[-1] not in ".!?…»":
            print("🟡 Ответ модели кажется незавершённым.")
        # Фильтр на повторяющийся ответ (во избежание зацикливания)
        if last_response and response.strip().lower().startswith(last_response.strip().lower()):
            print("🔁 Похоже на повтор (ответ) — пропускаю озвучивание...")
            last_text = user_text
            last_response = response
            continue
        # Печатаем и озвучиваем ответ
        print(f"Elaine-Сама: {response}")
        speak_text(response)
        # **Важно:** ждем небольшую паузу перед возобновлением прослушивания,
        # чтобы TTS-голос не был воспринят микрофоном
        sleep(1.0)
        # Обновляем историю диалога (храним последние несколько сообщений)
        # Формируем запись для истории диалога с использованием имени пользователя из llm
        entry = f"{USER_NAME}: {user_text}\nЭлейн-Сама: {response}"
        if entry not in chat_history:
            chat_history.append(entry)
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]
        # Обновляем последние реплики
        last_text = user_text
        last_response = response

if __name__ == "__main__":
    main()
