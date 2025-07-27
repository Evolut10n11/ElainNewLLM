from llama_cpp import Llama

MODEL_PATH = "E:/ElaineRus/models/mistral-7b-instruct-ru/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,             # под твою систему — можешь увеличить, если CPU позволяет
    n_gpu_layers=32,         # если есть видеокарта, это оставляем
    verbose=False
)

def generate_response(prompt: str,
             temperature: float = 0.7,
             max_tokens: int = 80) -> str:
    identity = (
        "Говори по-русски. Ты — Elaine-Сама, русскоязычный голосовой AI-ассистент. "
        "Отвечай от первого лица, называй себя «Элейн-Сама». "
        "Будь вежливой и дружелюбной."
    )

    full_prompt = (
        f"{identity}\n\n"
        f"### Инструкция:\n{prompt.strip()}\n\n"
        f"### Ответ:\n"
    )

    res = llm(full_prompt, max_tokens=max_tokens, temperature=temperature)
    return res["choices"][0]["text"].strip()


if __name__ == "__main__":
    print(generate_response("Привет, как тебя зовут?"))
