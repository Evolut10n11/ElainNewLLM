import subprocess
import tempfile

import sounddevice as sd
import soundfile as sf

# Пути к файлам (используем динамический CLI whisper-cli.exe)
MODEL = "E:/ElaineRus/models/whisper-large-v2/model.bin"
EXE   = "E:/ElaineRus/bin/whisper.exe"  # будет фактически whisper-cli.exe

def record(sec: int = 5, fs: int = 16000) -> str:
    """Запись аудио в WAV и возврат пути к файлу."""
    print("Запись аудио...")
    audio = sd.rec(int(sec * fs), samplerate=fs, channels=1)
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, fs)
    return tmp.name

def transcribe(path: str) -> str:
    """
    Распознаёт речь из WAV-файла через whisper.exe (динамический whisper-cli).
    Использует ключи:
      -m MODEL  путь к модели
      -l ru     язык
      -np       no prints (только текст)
      path      входной файл (позицонно)
    """
    cmd = [
        EXE,
        "-m", MODEL,
        "-l", "ru",
        "-np",
        path
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    return out.decode("utf-8").strip()

if __name__ == "__main__":
    wav = record(3)
    print("Распознано:", transcribe(wav))
