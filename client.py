import json
from time import sleep

from PySide6.QtCore import QObject
from PySide6.QtWebSockets import QWebSocket


class Manager(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.websocket = QWebSocket()
        self.websocket.textMessageReceived.connect(self.handle_text_message_received)
        self.websocket.connected.connect(self.handle_connected)
        self.test()

    def start(self):
        url = "ws://localhost:28765"
        self.websocket.open(url)

    def test(self):
        print("11")

        self.websocket.sendTextMessage(
            json.dumps({"method": "SUBSCRIBE", "params": ["btcusdt@aggTrade"], "id": 2}))

    def subscribe(self):
        info = {"method": "SUBSCRIBE", "params": ["wwwwwwwwww"], "id": 1}
        message = json.dumps(info)
        ret = self.websocket.sendTextMessage(message)
        assert ret == len(message)

    def handle_connected(self):
        self.subscribe()

    def handle_text_message_received(self, message):
        data = json.loads(message)
        print(data)


def main():
    import sys
    from PySide6.QtCore import QCoreApplication

    app = QCoreApplication(sys.argv)
    manager = Manager()
    manager.start()

    app.exec()


if __name__ == "__main__":
    main()
