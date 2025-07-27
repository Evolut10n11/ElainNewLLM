# ElainNewLLM

## Описание
Проект объединяет три сервиса:
- **STT**: преобразование речи в текст (Whisper.cpp).
- **LLM**: генерация ответов (llama.cpp с моделью Mistral).
- **TTS**: озвучка ответов (pyttsx3 с голосом Irina или любым WAV).

## Установка

1. Клонировать репозиторий:
   ```bash
   git clone https://github.com/Evolut10n11/ElainNewLLM.git
   cd ElainNewLLM
   ```

2. Создать и активировать виртуальное окружение:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   source .venv/bin/activate  # Linux/macOS
   ```

3. Установить зависимости:
   ```bash
   pip install -r requirements.txt
   ```

4. Сконфигурировать пути в `services/stt.py`, `services/llm.py`, `services/tts.py`:
   - Указать пути к `whisper.exe`, модели Whisper.
   - Указать пути к `llama-run.exe`, модели Mistral.
   - Настроить `pyttsx3` для голоса Irina или путь к своему WAV.

## Запуск

```bash
python main.py
```

Нажмите и удерживайте **F4** для записи голоса. После записи бот автоматически распознает речь, сгенерирует ответ и озвучит его.

## Свой WAV-голос

Чтобы использовать свой WAV в качестве голоса:
1. Поместите ваш файл `voice.wav` в папку `voice_samples/`.
2. В `services/tts.py` замените `VOICE_SAMPLE` на путь к вашему файлу:
   ```python
   VOICE_SAMPLE = os.path.join(os.getcwd(), "voice_samples", "voice.wav")
   ```

## Лицензия
MIT
