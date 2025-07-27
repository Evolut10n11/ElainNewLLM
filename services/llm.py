from llama_cpp import Llama

MODEL_PATH = "E:/ElaineRus/models/mistral-7b-instruct-ru/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=32,
    verbose=False
)

def generate_response(prompt: str,
                      history: list[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 300) -> str:
    identity = (
        "Говори по-русски. Ты — Elaine-Сама, русскоязычный голосовой AI-ассистент. "
        "Отвечай от первого лица, называй себя «Элейн-Сама». "
        "Будь вежливой и дружелюбной."
    )

    history_prompt = "\n".join(history or [])
    full_prompt = f"{identity}\n\n{history_prompt}\nТы: {prompt.strip()}\nЭлейн-Сама:"

    res = llm(full_prompt, max_tokens=max_tokens, temperature=temperature)
    return res["choices"][0]["text"].strip()