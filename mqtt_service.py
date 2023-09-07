import socket

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = '127.0.0.1'
port = 8888

server_socket.bind((host, port))

server_socket.listen(5)
print('服务器启动,等待客户端连接...')

while True:
    client_socket, addr = server_socket.accept()
    print('连接地址:', addr)

    while True:
        recv_data = client_socket.recv(1024)
        if len(recv_data) > 0:
            print('接收到的数据:', str(recv_data))
        else:
            print('客户端关闭')
            break

    # client_socket.close()

server_socket.close()