# ElaineLite – Русскоязычный голосовой AI-ассистент

Локальный AI-компаньон, работающий без интернета. Использует:

- 🎙 `whisper.cpp` — распознавание речи (STT)
- 🧠 `mistral-7b-instruct` через `llama.cpp` — генерация ответов
- 🔊 `Silero TTS` — озвучка
- 💾 `chat_history` — кратковременная память
- 📁 `output/last.txt` — экспорт текста для OBS / VTube Studio

## Запуск
1. Установи Python 3.11+
2. Установи зависимости:
```
pip install -r requirements.txt
```
3. Запусти:
```
python main.py
```

## Структура проекта

```
ElaineLite/
├── main.py
├── services/
│   ├── stt_vad.py       # Whisper + фильтрация
│   ├── llm.py           # llama.cpp-интерфейс
│   └── tts_silero.py    # Silero голос
├── output/              # Аудио и last.txt
└── models/              # Модели: Whisper, Mistral, Silero
```

## Требования
- Windows 11
- CUDA 12.9 (опционально)
- whisper.cpp + llama.cpp собраны вручную
- ffmpeg.exe в `PATH`

---
Проект собирался с ❤️ Ваней и Элейн-Сама.
