import asyncio
import time

from services.stt_vad    import record_vad, transcribe_vad
from services.llm       import generate_response
from services.tts_silero import speak_text
from vtube_controller   import vts_client

def main():
    # 1) Аутентификация VTube Studio — один раз при старте
    try:
        asyncio.run(vts_client.authenticate())
    except Exception as e:
        print(f"❌ Аутентификация не удалась: {e}")
        return

    chat_history  = []
    last_response = ""

    while True:
        wav = record_vad()
        if not wav:
            continue

        user_text = transcribe_vad(wav)
        if not user_text.strip():
            print("😶 Ничего не распознано. Попробуйте ещё раз.")
            continue

        print(f"Вы сказали: {user_text}")

        # 2) Фильтрация эха: если это снова наш последний ответ — пропускаем
        if last_response and user_text.lower().startswith(last_response.lower()):
            print("🔁 Эхо TTS — пропускаю...")
            continue

        # 3) Генерируем ответ через LLM
        response = generate_response(user_text, chat_history)

        # 4) Если нет завершающего знака — добавляем точку
        if response and response[-1] not in ".!?…»":
            response += "."

        print(f"Elaine-Сама: {response}")

        # 5) Озвучиваем с синхронизацией рта
        duration = speak_text(response)

        # 6) Обновляем историю
        entry = f"Ты: {user_text}\nЭлейн-Сама: {response}"
        chat_history.append(entry)
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]
        last_response = response

        # 7) Сохраняем последний ответ
        with open("output/last.txt", "w", encoding="utf-8") as f:
            f.write(response)

        # Небольшая пауза перед новой записью
        time.sleep(0.3)

if __name__ == "__main__":
    main()
