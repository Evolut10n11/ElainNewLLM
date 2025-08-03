import os
from llama_cpp import Llama

# Путь к локальной модели LLaMA/Mistral
MODEL_PATH = "E:/ElaineRus/models/mistral-7b-instruct-ru/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=8192,
    n_threads=os.cpu_count(),   # задействуем все ядра CPU
    n_gpu_layers=100,            # часть слоёв на GPU
    verbose=False,
    use_mmap=True                # ускоренное чтение модели через mmap
)

# Имя пользователя, с которым общается ассистент. Если нужно, вынесите в конфиг.
USER_NAME = "Ваня"

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
    temperature: float = 0.7,
    max_tokens: int = 100,
) -> str:
    """
    Генерирует ответ модели на заданный запрос с учётом истории.
    Стоп-фразы предотвращают продолжение своей же речи.
    """
    identity = (
        "Говори по-русски. Ты — Elaine-Сама, русскоязычный голосовой AI-ассистент. "
        f"Ты общаешься с пользователем по имени {USER_NAME}. "
        "Отвечай от первого лица, называй себя «Элейн-Сама». "
        "Будь вежливой и дружелюбной. "
        "После своей реплики не продолжай диалог — не генерируй ответ за пользователя."
    )
    history_prompt = "\n".join(history or [])
    full_prompt = (
        f"{identity}\n\n{history_prompt}"
        f"\n{USER_NAME}: {prompt.strip()}\nЭлейн-Сама:"
    )
    stop_words = [
        f"\n{USER_NAME}:",
        "\nТы:",
        "\nЭлейн-Сама:",
    ]

    # Убираем n_batch — его llama_cpp не поддерживает в версии у вас
    res = llm(
        full_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop_words
    )
    return clean_response(res["choices"][0]["text"].strip())
