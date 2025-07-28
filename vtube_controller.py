import asyncio
import json
import websockets
import os

# Имя файла с токеном (ровно как у вас на диске)
TOKEN_FILE   = "vtubeStudio_token"
HOST         = "ws://localhost:8001"
PLUGIN_NAME  = "Elaine1"
PLUGIN_DEV   = "nvm1"

class VTubeStudioClient:
    def __init__(self):
        self.token = None

    async def _connect(self):
        try:
            return await websockets.connect(HOST)
        except Exception as e:
            print(f"❌ Не удалось подключиться к VTube Studio по {HOST}: {e}")
            return None

    async def authenticate(self):
        # 1) Получаем или запрашиваем токен
        if not os.path.exists(TOKEN_FILE):
            ws = await self._connect()
            if not ws: return
            await ws.send(json.dumps({
                "apiName":"VTubeStudioPublicAPI",
                "apiVersion":"1.0",
                "requestID":"requestToken",
                "messageType":"AuthenticationTokenRequest",
                "data":{
                    "pluginName":PLUGIN_NAME,
                    "pluginDeveloper":PLUGIN_DEV
                }
            }))
            res = json.loads(await ws.recv())
            self.token = res["data"]["authenticationToken"]
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(self.token)
            await ws.close()
        else:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                self.token = f.read().strip()

        # 2) Ждём, пока вы нажмёте Allow в VTube Studio
        print("🔐 Теперь в VTube Studio нажмите «Allow», потом Enter…")
        input()

        # 3) Отправляем сам запрос аутентификации
        ws = await self._connect()
        if not ws: return
        await ws.send(json.dumps({
            "apiName":"VTubeStudioPublicAPI",
            "apiVersion":"1.0",
            "requestID":"auth",
            "messageType":"AuthenticationRequest",
            "data":{
                "pluginName":PLUGIN_NAME,
                "pluginDeveloper":PLUGIN_DEV,
                "authenticationToken":self.token
            }
        }))
        auth_res = json.loads(await ws.recv())
        await ws.close()
        if not auth_res["data"].get("authenticated"):
            raise RuntimeError("❌ Auth failed: " + auth_res["data"].get("reason",""))
        print("✅ Аутентификация прошла успешно.")

    async def set_mouth_open(self, value: float):
        # Для каждого движения рта открываем новое WS-соединение
        ws = await self._connect()
        if not ws: return
        await ws.send(json.dumps({
            "apiName":"VTubeStudioPublicAPI",
            "apiVersion":"1.0",
            "requestID":"mouthMove",
            "messageType":"InjectParameterDataRequest",
            "data":{
                "pluginName":PLUGIN_NAME,
                "pluginDeveloper":PLUGIN_DEV,
                "authenticationToken":self.token,
                "parameterValues":[{"id":"MouthOpen","value":value}]
            }
        }))
        try:
            await ws.recv()  # ждём ACK
        except:
            pass
        finally:
            await ws.close()

# Единый клиент, который импортируем в main.py и tts_silero.py
vts_client = VTubeStudioClient()
