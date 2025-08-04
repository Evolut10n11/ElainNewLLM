import asyncio
import random
import time
from typing import Optional

# Импорт генератора ответа, TTS и описания экрана
try:
    from .llm import generate_response, USER_NAME  # type: ignore
    from .tts_silero import speak_text  # type: ignore
    from .screen_capture import describe_screen  # type: ignore
except ImportError:
    from llm import generate_response, USER_NAME  # type: ignore
    from tts_silero import speak_text  # type: ignore
    from screen_capture import describe_screen  # type: ignore

try:
    import mss  # type: ignore
    from PIL import Image  # type: ignore
except ImportError:
    mss = None  # type: ignore
    Image = None  # type: ignore


class AutoThinker:
    """
    Фоновая сущность, которая следит за периодами тишины и генерирует
    самостоятельные мысли. Теперь при отсутствии голоса ассистент
    дополнительно анализирует содержимое экрана: делает одиночный
    скриншот и включает его описание в подсказку для LLM.

    chat_history – общая история диалога, которая может использоваться
    при генерации ответов. silent_timeout – сколько секунд после
    последней активности должно пройти, чтобы ассистент заговорил сам.
    """

    def __init__(self, chat_history: list[str], silent_timeout: float = 15.0):
        self.chat_history = chat_history
        self.silent_timeout = silent_timeout
        self.last_voice_time = asyncio.get_event_loop().time()
        self.enabled = True
        self.chat_messages: list[str] = []

    def notify_voice_activity(self) -> None:
        """Вызывается при голосовой активности пользователя"""
        self.last_voice_time = asyncio.get_event_loop().time()

    def push_chat(self, message: str) -> None:
        """Добавляет новое сообщение из чата Twitch"""
        if message:
            self.chat_messages.append(message)
            if len(self.chat_messages) > 20:
                self.chat_messages = self.chat_messages[-20:]

    async def run(self) -> None:
        """Фоновая задача, генерирующая мысли в тишине"""
        loop = asyncio.get_event_loop()
        while self.enabled:
            await asyncio.sleep(2.0)
            now = loop.time()
            if now - self.last_voice_time < self.silent_timeout:
                continue  # голос недавно был — ждём

            prompt = self.build_prompt()
            print(" Тишина... Думаю сама:", prompt)

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

    def get_screen_summary(self) -> str:
        """
        Делает одиночный скриншот и возвращает краткое описание содержимого
        экрана с использованием функции describe_screen. Если библиотека mss
        или PIL недоступна, возвращает пустую строку.
        """
        if mss is None or Image is None:
            return ""
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
            summary = describe_screen(img)
            return summary
        except Exception:
            return ""

    def build_prompt(self) -> str:
        """
        Формирует запрос для генерации ответа. Если имеются свежие
        сообщения из чата, выбирает одно из последних. Если чат пуст,
        делает снимок экрана и строит запрос исходя из содержимого
        монитора. В противном случае просит ассистента сказать что-нибудь
        самой.
        """
        # Приоритет: используем последние сообщения из чата Twitch
        if self.chat_messages:
            sample = random.choice(self.chat_messages[-5:])
            return f"В чате написали: '{sample}'. Ответь как vtuber девушка, весело."
        # Если чат пуст, пробуем описать экран
        summary = self.get_screen_summary()
        if summary:
            return (
                f"На экране сейчас: {summary}. Поделись своим мнением, наблюдением "
                f"или дай совет, будто ты vtuber девушка, чтобы разговор был интересным."
            )
        # Если ни чата, ни экрана — говорим что-нибудь сама
        return "Скажи что-нибудь самой — шутку, мнение или фразу как vtuber девушка."