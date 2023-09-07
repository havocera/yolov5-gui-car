import asyncio
import json

import websockets

# 存储客户端连接的字典
clients = {}

async def main(websocket, path):
    # 存储WebSocket连接对象，以便在其他函数中使用
    clients[websocket] = None

    try:
        async for message in websocket:
            # 处理从客户端接收的消息
            print(f"Received message: {message}")
            info = {"class":"car","id": 1}
            message1 = json.dumps(info)
            await websocket.send(message1)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        # 在连接关闭时移除该连接
        del clients[websocket]

# 函数用于向特定客户端发送消息
async def send_message(client, message):
    try:
        await client.send(message)
    except websockets.exceptions.ConnectionClosed:
        pass

# 一个示例函数，向所有客户端广播消息
async def broadcast_message(message):
    for client in clients.keys():
        await send_message(client, message)

async def main_loop():
    while True:
        message = input("Enter a message to broadcast (or 'exit' to quit): ")

        if message == 'exit':
            break

        # 广播消息给所有客户端
        await broadcast_message(message)

# 创建WebSocket服务器
start_server = websockets.serve(main, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
