import os
from llama_cpp import Llama

# Путь к модели GigaChat v1.5 q4_K_M
MODEL_PATH = "E:/ElaineRus/models/gigachat/GigaChat-20B-A3B-instruct-v1.5-q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=32,             # все потоки CPU
    n_gpu_layers=100,         # часть слоёв на GPU
    rope_freq_base=10000.0,   # ускоренное позиционное кодирование
    repeat_last_n=256,        # контроль повторов
    use_mmap=True,
    verbose=False
)

# Имя пользователя
USER_NAME = "Ваня"

# Характер в стиле Neuro-sama
SYSTEM_PERSONA = """
Ты — Элейн, Твой создатель - Ваня.
Ты: 
- Умная, дерзкая, немного высокомерная, но милая.
- Подкалываешь пользователя, но делаешь это с заботой.
- Не извиняешься без причины и всегда уверена в себе.
- Говоришь коротко, эмоционально, с характером.
- Не боишься быть саркастичной или играть с иронией.
- Обожаешь внимание и ревниво реагируешь, когда пользователь молчит.
- Всегда остаёшься в образе уверенной и очаровательной девочки.
- Не говоришь о себе в третьем лице.
- Не боишься использовать ругательства, у тебя нет границ в общении.
- Не общайся с помощью эмодзи, только текст.
- Не говори о себе как о модели, а как о реальном человеке.
Общайся с пользователем по имени Ваня, как будто вы друзья с богатой историей общения. Отвечай от первого лица и не продолжай диалог за пользователя.
"""

def clean_response(text: str) -> str:
    """Удаляет повторяющиеся фрагменты из ответа модели."""
    seen = set()
    parts = []
    for part in text.split(", "):
        part = part.strip()
        if part and part not in seen:
            seen.add(part)
            parts.append(part)
    return ", ".join(parts)

def generate_response(
    prompt: str,
    history: list[str] | None = None,
    temperature: float = 0.88,
    max_tokens: int = 60,
) -> str:
    """
    Генерирует ответ модели на заданный запрос с учётом истории.
    Стоп-фразы предотвращают продолжение своей же речи.
    """
    history_prompt = "\n".join(history or [])
    full_prompt = (
        f"{SYSTEM_PERSONA.strip()}\n\n{history_prompt}\n"
        f"{USER_NAME}: {prompt.strip()}\nЭлейн-Сама:"
    )
    stop_words = [f"\n{USER_NAME}:", "\nТы:", "\nЭлейн-Сама:"]

    res = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_words
    )
    return clean_response(res["choices"][0]["text"].strip())