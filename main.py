from time import sleep
import threading

try:
    from services.stt_vad import record_vad, transcribe_vad
    from services.llm import generate_response, USER_NAME
    from services.tts_silero import speak_text
except ImportError:
    from stt_vad import record_vad, transcribe_vad  # type: ignore
    from llm import generate_response, USER_NAME  # type: ignore
    from tts_silero import speak_text  # type: ignore

from twitchio.ext import commands

CLIENT_ID = 'wytz41znebdbonzo66ospr4knl39j7'
CLIENT_SECRET = 'b95jtu9b4ibf1x4qsb27xxnkd76ivi'
TWITCH_TOKEN = 'si14jnonj102h4rvzyuk7hyhu71vk7'
TWITCH_NICK = 'kakoitochelikhihi'
TWITCH_CHANNEL = 'kakoitochelikhihi'
BOT_ID = '76417315'

class ElaineTwitchBot(commands.Bot):
    def __init__(self):
        super().__init__(
            token=TWITCH_TOKEN,
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            bot_id=BOT_ID,
            prefix='!',
            initial_channels=[TWITCH_CHANNEL],
        )

    async def event_ready(self):
        print("🟣 Twitch-бот готов к приёму сообщений.")

    async def event_message(self, message):
        print("📩 event_message вызван")

        if message.echo:
            return

        author = message.author.name
        content = message.content
        print(f"{author}: {content}")

        prompt = f"{USER_NAME}, в чате Twitch {author} написал: {content}. Ответь коротко и дружелюбно."
        response = generate_response(prompt, [])

        await message.channel.send(response)
        print(f"Элейн-Сама (чат): {response}")
        speak_text(response)

def run_twitch_bot():
    bot = ElaineTwitchBot()
    bot.run(with_adapter=True)

def main():
    last_text = None
    last_response = None
    chat_history = []

    try:
        from screen_capture import run_screen_observer  # type: ignore
        screen_thread = threading.Thread(target=run_screen_observer, args=(30.0,), daemon=True)
        screen_thread.start()
        print("🖥️ Запущено фоновое наблюдение за экраном.")
    except Exception:
        print("⚠️ Не удалось запустить наблюдение за экраном. Продолжаем работу без него.")

    twitch_thread = threading.Thread(target=run_twitch_bot, daemon=True)
    twitch_thread.start()
    print("🟣 Запущен Twitch-бот.")

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

        response = generate_response(user_text, chat_history)

        if user_text.strip()[-1] not in ".!?…»":
            print("🟡 Похоже, ты не договорил — не добавляю запрос в историю.")
        if response.strip()[-1] not in ".!?…»":
            print("🟡 Ответ модели кажется незавершённым.")
        if last_response and response.strip().lower().startswith(last_response.strip().lower()):
            print("🔁 Похоже на повтор (ответ) — пропускаю озвучивание...")
            last_text = user_text
            last_response = response
            continue

        print(f"Elaine-Сама: {response}")
        speak_text(response)
        sleep(1.0)

        entry = f"{USER_NAME}: {user_text}\nЭлейн-Сама: {response}"
        if entry not in chat_history:
            chat_history.append(entry)
        if len(chat_history) > 4:
            chat_history = chat_history[-4:]

        last_text = user_text
        last_response = response

if __name__ == "__main__":
    main()
