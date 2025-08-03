import asyncio
import random
from services.llm import generate_response, USER_NAME
from services.tts_silero import speak_text

class AutoThinker:
    def __init__(self, chat_history: list[str], silent_timeout: float = 15.0):
        self.chat_history = chat_history
        self.silent_timeout = silent_timeout
        self.last_voice_time = asyncio.get_event_loop().time()
        self.enabled = True
        self.chat_messages: list[str] = []

    def notify_voice_activity(self):
        """Вызывается при голосовой активности пользователя"""
        self.last_voice_time = asyncio.get_event_loop().time()

    def push_chat(self, message: str):
        """Добавляет новое сообщение из чата Twitch"""
        if message:
            self.chat_messages.append(message)
            if len(self.chat_messages) > 20:
                self.chat_messages = self.chat_messages[-20:]

    async def run(self):
        """Фоновая задача, генерирующая мысли в тишине"""
        loop = asyncio.get_event_loop()
        while self.enabled:
            await asyncio.sleep(2.0)
            now = loop.time()
            if now - self.last_voice_time < self.silent_timeout:
                continue  # голос недавно был — ждём

            prompt = self.build_prompt()
            print("🧠 Тишина... Думаю сама:", prompt)

            response = await loop.run_in_executor(None, generate_response, prompt, self.chat_history)
            if response.strip():
                await loop.run_in_executor(None, speak_text, response)

                # Обновляем историю
                entry = f"Элейн-Сама: {response}"
                self.chat_history.append(entry)
                if len(self.chat_history) > 6:
                    self.chat_history[:] = self.chat_history[-6:]

                # Сброс таймера, чтобы не говорить подряд
                self.last_voice_time = loop.time()

    def build_prompt(self) -> str:
        """Формирует запрос для генерации ответа"""
        if self.chat_messages:
            sample = random.choice(self.chat_messages[-5:])
            return f"В чате написали: '{sample}'. Ответь как vtuber девушка, весело."
        else:
            return "Скажи что-нибудь самой — шутку, мнение или фразу как vtuber девушка."
