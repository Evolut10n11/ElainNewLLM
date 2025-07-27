from services.stt_vad import record_vad, transcribe_vad
from services.llm import generate_response
from services.tts_silero import speak_text

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

        if last_text and user_text.strip().lower().startswith(last_text.strip().lower()):
            print("🔁 Похоже на повтор (вопрос) — пропускаю...")
            continue

        if user_text.strip()[-1] not in ".!?…»":
            print("🟡 Похоже, ты не договорил — не добавляю в историю.")
            response = generate_response(user_text, chat_history)
            print(f"Elaine-Сама: {response}")
            speak_text(response)
            with open("output/last.txt", "w", encoding="utf-8") as f:
                f.write(response)
            last_text = user_text
            last_response = response
            continue

        response = generate_response(user_text, chat_history)

        if last_response and response.strip().lower().startswith(last_response.strip().lower()):
            print("🔁 Похоже на повтор (ответ) — пропускаю...")
            continue

        last_text = user_text
        last_response = response

        entry = f"Ты: {user_text}\nЭлейн-Сама: {response}"
        if entry not in chat_history:
            chat_history.append(entry)
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]

        print(f"Elaine-Сама: {response}")
        speak_text(response)

        with open("output/last.txt", "w", encoding="utf-8") as f:
            f.write(response)

if __name__ == "__main__":
    main()
