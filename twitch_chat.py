# twitch_chat.py
import asyncio
from twitchio.ext import commands
from llm import generate_response, USER_NAME
from tts_silero import speak_text

# Замените на название вашего канала
CHANNEL_NAME = "kakoitochelikhihiа"

class ChatReader(commands.Bot):
    def __init__(self):
        super().__init__(
            token=None,  # без токена — режим read-only
            initial_channels=[CHANNEL_NAME],
            nick="kakoitochelikhihi",  # псевдоанонимные аккаунты Twitch
            prefix="!"
        )

    async def event_message(self, message):
        # Игнорируем собственные сообщения (если вдруг будет токен)
        if message.echo:
            return
        user = message.author.display_name
        text = message.content
        print(f"[Twitch] {user}: {text}")
        # Формируем ответ. В history можно включить последние сообщения чата или текущий history голосового диалога
        reply = generate_response(f"{user}: {text}", [])
        speak_text(reply)

def run_chat_listener():
    bot = ChatReader()
    bot.run()
