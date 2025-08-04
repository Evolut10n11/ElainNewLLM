"""
Модуль для взаимодействия с VTube Studio Public API.

Новая реализация обеспечивает:
* Постоянное WebSocket‑соединение и фоновую задачу «keep‑alive»,
  чтобы API не отключало плагин между вызовами.
* Обёртки `authenticate()` и `set_mouth_open()` доступны как
  синхронные методы. Они ставят задания в фоновую asyncio‑петлю,
  поэтому могут вызываться из любого потока без ожидания loop.run().
* Аутентификация срабатывает только при первом успешном подключении.

Использование:

```python
vts_client = VTubeStudioClient()
vts_client.authenticate()  # однажды, затем соединение поддерживается
vts_client.set_mouth_open(0.5)
```

Периодический keep‑alive автоматически отправляет запрос
`APIAvailableRequest` каждые 10 секунд, чтобы удерживать соединение.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any, Dict

import websockets  # type: ignore

TOKEN_FILE = "vtubeStudio_token.txt"
HOST = "ws://localhost:8001"
PLUGIN_NAME = "Elaine1"
PLUGIN_DEV = "nvm1"


class VTubeStudioClient:
    """Клиент для работы с VTube Studio Public API с постоянным соединением."""

    def __init__(self) -> None:
        # Переменные состояния
        self.token: str | None = None
        self.ws: websockets.WebSocketClientProtocol | None = None
        self.authenticated: bool = False
        self._printed_auth_success: bool = False
        # Фоновый asyncio‑цикл, работающий в отдельном потоке
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        # Задача keep‑alive
        self._keep_alive_task: asyncio.Future | None = None

    # ------------------------------------------------------------------
    # Вспомогательные асинхронные методы (исполняются в фоновом цикле)
    async def _connect(self) -> None:
        """Создаёт WebSocket‑подключение, если его нет или оно закрыто."""
        if self.ws is not None and not self.ws.closed:
            return
        self.ws = await websockets.connect(HOST)

    async def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Отправляет payload по WebSocket и получает ответ."""
        if self.ws is None or self.ws.closed:
            await self._connect()
        assert self.ws is not None
        await self.ws.send(json.dumps(payload))
        resp = await self.ws.recv()
        return json.loads(resp)

    async def _authenticate(self) -> None:
        """Асинхронная часть аутентификации."""
        # Убедимся, что есть соединение
        await self._connect()
        # Уже аутентифицированы
        if self.authenticated and self.ws is not None and not self.ws.closed:
            return
        # Загружаем или получаем токен
        token: str | None = None
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                token = f.read().strip()
        except FileNotFoundError:
            token = None
        if not token:
            req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "requestToken",
                "messageType": "AuthenticationTokenRequest",
                "data": {
                    "pluginName": PLUGIN_NAME,
                    "pluginDeveloper": PLUGIN_DEV,
                },
            }
            res = await self._send(req)
            token = res["data"]["authenticationToken"]
            # Сохраняем токен
            try:
                with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                    f.write(token)
            except OSError:
                pass
            # Пользователь должен подтвердить доступ в VTube Studio
            print("🔐 Теперь в VTube Studio нажмите ‘Allow’, потом нажмите Enter…")
            try:
                input()
            except EOFError:
                pass
        self.token = token
        # Отправляем запрос аутентификации
        auth_req = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "auth",
            "messageType": "AuthenticationRequest",
            "data": {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEV,
                "authenticationToken": self.token,
            },
        }
        auth_res = await self._send(auth_req)
        if not auth_res.get("data", {}).get("authenticated"):
            self.authenticated = False
            raise RuntimeError(
                "❌ Auth failed: " + auth_res.get("data", {}).get("reason", "")
            )
        self.authenticated = True
        if not self._printed_auth_success:
            print("✅ Аутентификация прошла успешно.")
            self._printed_auth_success = True
        # Запускаем keep‑alive, если он ещё не запущен
        if self._keep_alive_task is None:
            self._keep_alive_task = asyncio.ensure_future(self._keep_alive(), loop=self._loop)

    async def _keep_alive(self) -> None:
        """Периодически отправляет запрос APIAvailableRequest для поддержания соединения."""
        while True:
            try:
                # Каждые 10 секунд
                await asyncio.sleep(10)
                await self._connect()
                if not self.authenticated:
                    await self._authenticate()
                ping = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "keepAlive",
                    "messageType": "APIAvailableRequest",
                }
                await self._send(ping)
            except Exception:
                # В случае ошибки сбрасываем флаг и пробуем снова на следующей итерации
                self.authenticated = False
                continue

    async def _set_mouth_open_async(self, value: float) -> None:
        """Асинхронно отправляет параметр MouthOpen."""
        await self._connect()
        if not self.authenticated:
            await self._authenticate()
        value = max(0.0, min(1.0, float(value)))
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": "mouthMove",
            "messageType": "InjectParameterDataRequest",
            "data": {
                "parameterValues": [
                    {"id": "MouthOpen", "value": value}
                ]
            },
        }
        try:
            await self._send(payload)
        except Exception:
            self.authenticated = False
            raise

    # ------------------------------------------------------------------
    # Публичные методы (синхронные), вызываемые из любого потока
    def authenticate(self) -> None:
        """Гарантирует аутентификацию. Возвращает после завершения."""
        fut = asyncio.run_coroutine_threadsafe(self._authenticate(), self._loop)
        return fut.result()

    def set_mouth_open(self, value: float) -> None:
        """Синхронно устанавливает параметр MouthOpen."""
        fut = asyncio.run_coroutine_threadsafe(
            self._set_mouth_open_async(value), self._loop
        )
        return fut.result()