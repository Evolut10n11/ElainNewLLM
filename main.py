import re
import time
import keyboard          # pip install keyboard
from services.stt import record, transcribe
from services.llm import generate
from services.tts import speak

def main():
    print("Нажмите и удерживайте F4 для записи…")
    while True:
        keyboard.wait('F4')
        start = time.time()

        wav = record(3)
        user_text = transcribe(wav)
        print(f"Вы сказали ({time.time()-start:.1f}s):", user_text)

        answer = generate(user_text)
        print("Elaine-Сама отвечает:", answer)

        # чистим таймкоды/метаданные
        clean = re.sub(r"\[.*?\]", "", answer).replace("### Инструкция:", "").strip()
        print("Готовим к озвучке:", clean)

        out_wav = speak(clean)
        print("Озвучено в:", out_wav)
        print("\n---\nНажмите и удерживайте F4 для следующей записи…")

if __name__ == "__main__":
    main()
