import asyncio
from vtube_controller import VTubeStudioClient
import time

async def test_mouth():
    client = VTubeStudioClient()
    # Соединяемся и аутентифицируемся
    await client.authenticate()
    # Открываем/закрываем рот 5 раз
    for i in range(5):
        print(f"Тест анимации рта: цикл {i+1}")
        await client.set_mouth_open(1.0)   # открыть рот
        time.sleep(0.5)                   # держим открытым 0.5 сек
        await client.set_mouth_open(0.0)   # закрыть рот
        time.sleep(0.5)
    print("✅ Тест мимики лица завершён.")

if __name__ == "__main__":
    asyncio.run(test_mouth())
