import asyncio
import json
import websockets
import os

TOKEN_FILE   = "vtubeStudio_token.txt"
HOST         = "ws://localhost:8001"
PLUGIN_NAME  = "Elaine1"
PLUGIN_DEV   = "nvm1"

class VTubeStudioClient:
    """
    Клиент для взаимодействия с VTube Studio Public API. Поддерживает
    повторное использование WebSocket‑соединения и выполняет
    аутентификацию только при первом вызове.
    """

    def __init__(self) -> None:
        # Сохранённый токен; читается из файла или запрашивается у VTS.
        self.token: str | None = None
        # Текущий WebSocket‑клиент. Может быть None, если ещё не соединены.
        self.ws: websockets.WebSocketClientProtocol | None = None
        # Флаг успешной аутентификации. Используется в паре с
        # authenticated_ws, чтобы определять, нужно ли повторно
        # аутентифицировать текущее соединение.
        self.authenticated: bool = False
        # Последнее WebSocket‑соединение, на котором выполнена
        # аутентификация. Если соединение меняется, требуется
        # повторная аутентификация.
        self._auth_ws: websockets.WebSocketClientProtocol | None = None

    async def connect(self) -> None:
        """
        Устанавливает соединение с WebSocket API VTube Studio, если его ещё нет
        или оно было закрыто. Подключение открывается только один раз.
        """
        # Если уже есть открытое соединение, ничего не делаем
        if self.ws is not None and not self.ws.closed:
            return
        # Создаём новое подключение
        self.ws = await websockets.connect(HOST)

    async def send(self, payload: dict) -> dict:
        """
        Отправляет запрос в VTube Studio и возвращает ответ в виде словаря.
        Предполагается, что соединение уже установлено.
        """
        assert self.ws is not None, "WebSocket connection is not established"
        await self.ws.send(json.dumps(payload))
        response = await self.ws.recv()
        return json.loads(response)

    async def authenticate(self) -> None:
        """
        Выполняет аутентификацию плагина в VTube Studio.
        Повторные вызовы безопасны: если клиент уже аутентифицирован,
        метод просто завершится.
        """
        # Убедимся, что есть подключение
        await self.connect()
        # Если текущее соединение уже было аутентифицировано, выходим
        if (
            self.authenticated
            and self.ws is not None
            and self.ws == self._auth_ws
            and not self.ws.closed
        ):
            return
        # Загрузка или получение токена
        if not os.path.exists(TOKEN_FILE):
            # Запрашиваем токен у VTS
            tok_req = {
                "apiName": "VTubeStudioPublicAPI",
                "apiVersion": "1.0",
                "requestID": "requestToken",
                "messageType": "AuthenticationTokenRequest",
                "data": {
                    "pluginName": PLUGIN_NAME,
                    "pluginDeveloper": PLUGIN_DEV,
                },
            }
            tok_res = await self.send(tok_req)
            self.token = tok_res["data"]["authenticationToken"]
            # Сохраняем токен для последующего использования
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                f.write(self.token)
            # Просим пользователя подтвердить доступ
            print("🔐 Теперь в VTube Studio нажмите «Allow», потом нажмите Enter…")
            try:
                input()
            except EOFError:
                # Игнорируем EOFError в автоматизированных тестах
                pass
        else:
            # Читаем сохранённый токен
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                self.token = f.read().strip()
        # Отправляем запрос аутентификации с токеном
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
        auth_res = await self.send(auth_req)
        if not auth_res["data"].get("authenticated"):
            raise RuntimeError("❌ Auth failed: " + auth_res["data"].get("reason", ""))
        # Отмечаем, что мы успешно аутентифицированы на текущем соединении
        self.authenticated = True
        self._auth_ws = self.ws
        print("✅ Аутентификация прошла успешно.")

    async def set_mouth_open(self, value: float) -> None:
        """
        Инжектирует значение входного параметра MouthOpen.
        Диапазон value: 0.0 (рот закрыт) ... 1.0 (рот полностью открыт).
        """
        # Убедимся, что соединение установлено и мы аутентифицированы
        await self.connect()
        # authenticate() проверит, требуется ли повторная аутентификация
        await self.authenticate()
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
        res = await self.send(payload)
        # Для отладки выводим результат (ожидается пустой data при успехе)
        #print(f"📤 InjectParameterDataResponse: {res}")
