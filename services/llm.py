from llama_cpp import Llama

MODEL_PATH = "E:/ElaineRus/models/mistral-7b-instruct-ru/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,
    n_threads=8,
    n_gpu_layers=32,
    verbose=False
)

def generate(prompt: str,
             temperature: float = 0.7,
             max_tokens: int = 128) -> str:
    identity = (
        "Ты — Elaine-Сама, русскоязычный голосовой AI-ассистент. "
        "Отвечай от первого лица, называй себя «Элейн-Сама». "
        "Будь вежливой и дружелюбной.\n\n"
    )
    full_prompt = (
        identity +
        "### Инструкция:\n"
        f"{prompt}\n\n"
        "### Ответ:\n"
    )
    res = llm(full_prompt, max_tokens=max_tokens, temperature=temperature)
    text = res["choices"][0]["text"]
    if "### Ответ:" in text:
        return text.split("### Ответ:")[-1].strip()
    return text.strip()

if __name__ == "__main__":
    print(generate("Привет, как тебя зовут?"))
