import asyncio
from vtube_controller import VTubeStudioClient

async def main():
    vts = VTubeStudioClient()
    await vts.authenticate()

    # Открываем рот на 1.5 секунды
    await vts.set_mouth_open(1.0)
    await asyncio.sleep(1.5)
    # Закрываем рот
    await vts.set_mouth_open(0.0)

asyncio.run(main())
