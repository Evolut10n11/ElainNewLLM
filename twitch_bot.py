from twitchio.ext import commands
from llm import generate_response, USER_NAME
from tts_silero import speak_text

# Настройки Twitch
TWITCH_TOKEN = 'si14jnonj102h4rvzyuk7hyhu71vk7'
TWITCH_NICK = 'kakoitochelikhihi'  # замените на точный никнейм вашего Twitch-аккаунта
TWITCH_CHANNEL = 'kakoitochelikhihi'  # канал, к которому подключаемся
CLIENT_ID = '9ro3fog9cavz8obw8tnuikn4u8gsed'
CLIENT_SECRET = 'ohvsovwwcecp5orjws5mk9d2hbtbcrb'

class ElaineTwitchBot(commands.Bot):

    def __init__(self):
        super().__init__(token=TWITCH_TOKEN, prefix='!', initial_channels=[TWITCH_CHANNEL])

    async def event_ready(self):
        print(f"🟣 Подключились к чату Twitch как {self.nick}")

    async def event_message(self, message):
        if message.echo:
            return

        author = message.author.name
        content = message.content
        print(f"{author}: {content}")

        # Генерируем ответ модели
        prompt = f"{USER_NAME}, в чате Twitch {author} написал: {content}. Ответь коротко и дружелюбно."
        response = generate_response(prompt, [])

        # Отвечаем в чате
        await message.channel.send(response)
        print(f"Элейн-Сама (чат): {response}")

        # Озвучиваем ответ
        speak_text(response)

bot = ElaineTwitchBot()
bot.run()