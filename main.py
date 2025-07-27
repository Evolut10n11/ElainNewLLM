from services.stt_vad import record_vad, transcribe_vad
from services.llm import generate_response
from services.tts_silero import speak_text

last_text = None

def main():
    while True:
        wav = record_vad()
        user_text = transcribe_vad(wav)

        if not user_text.strip():
            print("😶 Ничего не распознано. Попробуйте ещё раз.")
            continue

        print(f"Вы сказали: {user_text}")

        # Защита от повтора одного и того же ввода
        global last_text
        if user_text == last_text:
            print("🌀 Повторный ввод — пропускаю...")
            continue
        last_text = user_text

        response = generate_response(user_text)
        print(f"Elaine-Сама: {response}")
        speak_text(response)

if __name__ == "__main__":
    main()
