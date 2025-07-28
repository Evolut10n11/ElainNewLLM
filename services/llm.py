from llama_cpp import Llama

MODEL_PATH = "E:/ElaineRus/models/mistral-7b-instruct-ru/mistral-7b-instruct-v0.1.Q4_K_M.gguf"

llm = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=8, n_gpu_layers=32, verbose=False)

def clean_response(text):
    seen = set(); parts=[]
    for p in text.split(", "):
        p = p.strip()
        if p and p not in seen:
            seen.add(p); parts.append(p)
    return ", ".join(parts)

def generate_response(prompt: str, history: list[str]=None,
                      temperature:float=0.7, max_tokens:int=100) -> str:
    identity = (
        "Говори по-русски. Ты — Elaine-Сама, русскоязычный голосовой AI-ассистент женского пола, так что отвечай как девушка от лица девочки. "
        "Ты общаешься с пользователем по имени Ваня. "
        "Отвечай от первого лица, называй себя «Элейн-Сама». "
        "Будь вежливой и дружелюбной."
    )
    hist = "\n".join(history or [])
    prompt_full = f"{identity}\n\n{hist}\nТы: {prompt.strip()}\nЭлейн-Сама:"
    res = llm(prompt_full, max_tokens=max_tokens, temperature=temperature)
    return clean_response(res["choices"][0]["text"].strip())
