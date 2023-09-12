import sys
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtMultimedia import QMediaContent, QMediaPlayer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSizePolicy
from PySide6.QtMultimediaWidgets import QVideoWidget

class RTSPPlayerApp(QMainWindow):
    def __init__(self, rtsp_url):
        super().__init__()

        self.setWindowTitle("RTSP Player")
        self.setGeometry(100, 100, 800, 600)

        # Create a central widget to hold the video widget
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # Create a video widget
        self.video_widget = QVideoWidget()
        self.video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Create a media player
        self.media_player = QMediaPlayer()
        self.media_player.setVideoOutput(self.video_widget)

        # Load the RTSP stream
        media_content = QMediaContent(QUrl(rtsp_url))
        self.media_player.setMedia(media_content)

        # Create Play and Pause buttons
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")

        self.play_button.clicked.connect(self.play_video)
        self.pause_button.clicked.connect(self.pause_video)

        # Create a layout for the buttons
        button_layout = QVBoxLayout()
        button_layout.addWidget(self.play_button)
        button_layout.addWidget(self.pause_button)

        # Create a layout for the video widget and buttons
        layout = QVBoxLayout()
        layout.addWidget(self.video_widget)
        layout.addLayout(button_layout)

        central_widget.setLayout(layout)

    def play_video(self):
        self.media_player.play()

    def pause_video(self):
        self.media_player.pause()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    rtsp_url = "rtsp://your_rtsp_stream_url"
    player = RTSPPlayerApp(rtsp_url)
    player.show()
    sys.exit(app.exec_())
