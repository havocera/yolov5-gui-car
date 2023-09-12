# ///////////////////////////////////////////////////////////////
#
# ///////////////////////////////////////////////////////////////
import asyncio
import datetime
import glob
import json
import socket
import sys
import os
import platform
import threading
import time
from functools import partial

import numpy as np


from PySide6.QtCore import QTimer, Signal
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QMainWindow
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph

import cv2
import sys
import sys

import yolo_thread

# 里面替换为自己项目目录下的文件路径
sys.path.insert(0, os.path.abspath('.'))
import onnxruntime
from PySide6.QtWebSockets import QWebSocket
from modules.resources_rc import *

# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *

from widgets import *
import globals

os.environ["QT_FONT_DPI"] = "96"  # FIX Problem for High DPI and Scale above 100%

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None


class EmittingStr(QtCore.QObject):
    textWritten = QtCore.Signal(str)

    def write(self, text):
        self.textWritten.emit(str(text))
        loop = QEventLoop()
        QTimer.singleShot(100, loop.quit)
        loop.exec_()
        QApplication.processEvents()


class MainWindow(QMainWindow):
    resSignal = Signal(str)
    createVideoFinshSignal = Signal()

    def __init__(self, yoloModel):
        QMainWindow.__init__(self)
        self.car = None
        self.yolo = yoloModel
        sys.stdout = EmittingStr()
        sys.stderr = EmittingStr()
        sys.stdout.textWritten.connect(self.outputWritten)
        sys.stderr.textWritten.connect(self.outputWritten)
        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////

        self.capture = None
        self.capture3 = None
        self.capture2 = None
        self.capture1 = None
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        # self.fourcc = cv2.VideoWriter_fourcc('mp4v')
        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "赛车实况图像自动跟踪识别软件"
        description = "赛车实况图像自动跟踪识别软件"
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)
        self.strpath = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # QTableWidget PARAMETERS
        # ///////////////////////////////////////////////////////////////
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.timer = None
        self.timer1 = None
        self.timer2 = None
        self.timer3 = None
        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////
        # widgets.stackedWidget.addWidget(widgets.home_widget_3)
        # widgets.stackedWidget.addWidget(widgets.home_widget_4)
        # widgets.stackedWidget.addWidget(widgets.home_widget_5)
        # widgets.stackedWidget.addWidget(widgets.home_widget_6)
        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)
        widgets.btn_exit.clicked.connect(self.buttonClick)
        widgets.playTopBtn.clicked.connect(self.buttonClick)
        widgets.playTopBtn2.clicked.connect(self.yolo_contro)
        widgets.refreshbtn.clicked.connect(self.refreshcar)
        widgets.editbtn.clicked.connect(self.editcar)
        widgets.home_widget_3.DoubleClicked.connect(partial(self.current_wight, 1))
        widgets.home_widget_4.DoubleClicked.connect(partial(self.current_wight, 2))
        widgets.home_widget_5.DoubleClicked.connect(partial(self.current_wight, 4))
        widgets.home_widget_6.DoubleClicked.connect(partial(self.current_wight, 3))

        self.noopen = []
        self.labelSize = widgets.home_widget_5.frameSize()
        self.isStart = 1
        self.thread_ids = []
        self.current_video = 0

        self.link = []
        self.CLASSES = []
        self.refreshcar()

        # widgets.tableWidget.clear()
        self.base_path = os.path.abspath('.') + "/video/"

        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)

        # print(self.labelSize)

        # EXTRA LEFT BOX
        def openCloseLeftBox(self):
            UIFunctions.toggleLeftBox(self, True)

        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)

        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)
        # self.input_name = self.get_input_name()  # ['images']
        # self.output_name = self.get_output_name()  # ['output0']
        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme = False
        themeFile = "./themes/py_dracula_light.qss"
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;5000"
        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        self.isStopyolo = True
        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))
        self.getlinkTable()
        if len(self.link) != 0:
            self.defaultLink = self.link[0][1]
        else:
            self.defaultLink = ""
        self.link1 = self.defaultLink
        self.link2 = self.link[1][1]
        self.link3 = self.link[1][1]
        self.link4 = self.link[1][1]
        self.out1 = None
        self.out2 = None
        self.out3 = None
        self.out4 = None
        self.yolo.yoloSignl.connect(self.isclass)
        self.yolo.yoloStatusSignl.connect(self.yoloStatus)
        widgets.linkTable.setColumnCount(2)
        # widgets.playTopBtn2.connect(self.yolo_contro)
        # widgets.linkTable.setRowCount(3)
        widgets.linkTable.setHorizontalHeaderLabels(["序号", "url"])
        widgets.linkTable.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for i in range(len(self.link)):
            item = self.link[i]
            row = widgets.linkTable.rowCount()
            widgets.linkTable.insertRow(row)
            for j in range(len(item)):
                item = QTableWidgetItem(str(self.link[i][j]))
                item.setTextAlignment(Qt.AlignCenter)
                # item.setFlags(item.flags() != QtCore.Qt.ItemIsEditable)
                widgets.linkTable.setItem(row, j, item)
        widgets.creditsLabel.setText("软件已经启动")
        self.isyolo = True

    def yoloStatus(self, data):
        msg = json.loads(data)
        if msg["status"]:
            widgets.creditsLabel.setText(msg["message"])
        else:
            widgets.creditsLabel.setText("未知错误")

    def stop_yolo(self):
        self.yolo.isStopyolo = False
        thread_count = 0
        for i in self.yolo.thread_ids:
            i.join()
            if i.is_alive():
                thread_count += 1
                widgets.creditsLabel.setText(f"关闭线程{thread_count}")
        widgets.creditsLabel.setText(f"关闭线程完成，正在合成视频")
        for num in range(1, 5):
            self.yolo.recorders["stream" + str(num)].stop()
        self.yolo.createVideo()
        widgets.creditsLabel.setText(f"合成视频完成，在软件目录的video文件夹自取")

    def yolo_contro(self):

        if self.isyolo:
            self.startyolobox = QMessageBox.question(self, '提醒', '确定开启识别，并启动本次视频录制？',
                                                 QMessageBox.StandardButton.Ok,
                                                 QMessageBox.StandardButton.No)
            if self.startyolobox == QMessageBox.StandardButton.Ok:
                widgets.creditsLabel.setText("正在开启视频识别")
                s_thread = threading.Thread(target=self.yolo.captureMutipleCamera)
                s_thread.start()
                self.isyolo = False
        else:
            self.endyolobox = QMessageBox.question(self, '提醒', '确定关闭本次识别，并开始合成视频？',
                                                 QMessageBox.StandardButton.Ok,
                                                 QMessageBox.StandardButton.No)
            if self.endyolobox == QMessageBox.StandardButton.Ok:
                widgets.creditsLabel.setText("正在关闭视频识别")
                self.yolo.isStopyolo = False
                # thread_count = 0
                # for i in self.yolo.thread_ids:
                #     i.join()
                #     if i.is_alive():
                #         thread_count += 1
                #         widgets.creditsLabel.setText(f"关闭线程{thread_count}")
                #
                # for num in range(1, 5):
                #     self.yolo.recorders["stream" + str(num)].stop()
                # self.yolo.createVideo()
                stop_yolo_thread = threading.Thread(target=self.stop_yolo)
                stop_yolo_thread.start()
                widgets.creditsLabel.setText("正在子线程关闭视频识别")
                self.isyolo = True

    ######################################################
    #              双击放大功能实现                       #
    # ####################################################
    def current_wight(self, num):
        if num == 1:
            if self.current_video == 0:
                self.current_video = 1
                widgets.home_widget_3.setVisible(True)
                widgets.home_widget_4.setVisible(False)
                widgets.home_widget_5.setVisible(False)
                widgets.home_widget_6.setVisible(False)
                widgets.home_widget_2.setVisible(False)
            else:
                self.current_video = 0
                widgets.home_widget_3.setVisible(True)
                widgets.home_widget_4.setVisible(True)
                widgets.home_widget_5.setVisible(True)
                widgets.home_widget_6.setVisible(True)
                widgets.home_widget_2.setVisible(True)
        elif num == 2:
            if self.current_video == 0:
                self.current_video = 2
                widgets.home_widget_3.setVisible(False)
                widgets.home_widget_4.setVisible(True)
                widgets.home_widget_5.setVisible(False)
                widgets.home_widget_6.setVisible(False)
                widgets.home_widget_2.setVisible(False)
            else:
                self.current_video = 0
                widgets.home_widget_3.setVisible(True)
                widgets.home_widget_4.setVisible(True)
                widgets.home_widget_5.setVisible(True)
                widgets.home_widget_6.setVisible(True)
                widgets.home_widget_2.setVisible(True)
        elif num == 3:
            if self.current_video == 0:
                self.current_video = 3
                widgets.home_widget_3.setVisible(False)
                widgets.home_widget_4.setVisible(False)
                widgets.home_widget_5.setVisible(False)
                widgets.home_widget_6.setVisible(True)
                widgets.videoLayout.setVisible(False)
            else:
                self.current_video = 0
                widgets.home_widget_3.setVisible(True)
                widgets.home_widget_4.setVisible(True)
                widgets.home_widget_5.setVisible(True)
                widgets.home_widget_6.setVisible(True)
                widgets.videoLayout.setVisible(True)
        elif num == 4:
            if self.current_video == 0:
                self.current_video = 4
                # widgets.home_widget_3.setVisible(False)
                # widgets.home_widget_4.setVisible(False)
                widgets.home_widget_5.setVisible(True)
                widgets.home_widget_6.setVisible(False)
                widgets.videoLayout.setVisible(False)
            else:
                self.current_video = 0
                widgets.home_widget_3.setVisible(True)
                widgets.home_widget_4.setVisible(True)
                widgets.home_widget_5.setVisible(True)
                widgets.home_widget_6.setVisible(True)
                widgets.videoLayout.setVisible(True)

        print("shuangji", num)
        # widgets.stackedWidget.setCurrentWidget(wight_list[num])

    def outputWritten(self, text):
        with open("log.txt", "a",encoding="utf-8") as f:
            now =datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d,%H:%M:%S")

            f.write("-----------"+formatted_datetime +"-----------\n"+text + "\n")

        # widgets.creditsLabel.setText(text)

    def isclass(self, data):
        try:
            data = json.loads(data)
            print(data)
        except Exception as e:
            print(e)
        try:
            frame_num = self.car.index(data["class"])
        except ValueError:
            print(ValueError)
            return 0
        if frame_num == 1:
            if self.link1 != data["link"]:
                self.link1 = data["link"]
                self.capture.release()
                if self.timer == None:
                    self.start_capture(data["link"])
                else:

                    self.capture = None
                    # self.timer.stop()
                    self.start_capture(data["link"])

        elif frame_num == 2:
            if self.link2 != data["link"]:
                self.link2 = data["link"]
                self.capture1.release()
                if self.timer1 == None:
                    self.start_capture_1(data["link"])
                else:

                    self.capture1 = None
                    # self.timer1.stop()
                    self.start_capture_1(data["link"])
        elif frame_num == 3:
            if self.link3 != data["link"]:
                self.link3 = data["link"]
                self.capture2.release()
                if self.timer2 == None:
                    self.start_capture_2(data["link"])
                else:

                    self.capture2 = None
                    # self.timer2.stop()
                    self.start_capture_2(data["link"])
        elif frame_num == 4:
            if self.link4 != data["link"]:
                self.link4 = data["link"]
                self.capture3.release()
                if self.timer3 == None:
                    self.start_capture_3(data["link"])
                else:

                    self.capture3 = None
                    # self.timer3.stop()
                    self.start_capture_3(data["link"])

        # print(data)

    def getlinkTable(self):
        if os.path.exists("link.txt"):
            pass
        else:
            newfile = "link.txt"
            new_file = open(newfile, "w")  # 用open函数建立一个新的文本文件
            new_file.close()
        with open("./link.txt", 'r') as f:
            linkNum = 1
            for i in f.read().split("\n"):
                if i != "":
                    self.link.append([linkNum, i])
                    linkNum = linkNum + 1

    def refreshcar(self):
        if os.path.exists("car.txt"):
            pass
        else:
            newfile = "car.txt"
            new_file = open(newfile, "w")  # 用open函数建立一个新的文本文件
            new_file.close()
        widgets.comboBox_4.clear()
        widgets.comboBox_5.clear()
        widgets.comboBox_6.clear()
        widgets.comboBox_7.clear()

        with open("car.txt", "r") as f:
            cardata = f.read().split("\n")
            self.CLASSES = cardata
            for i in cardata:
                widgets.comboBox_4.addItem(i)
                widgets.comboBox_5.addItem(i)
                widgets.comboBox_6.addItem(i)
                widgets.comboBox_7.addItem(i)

    def editcar(self):
        if os.path.exists("car.txt"):
            pass
        else:
            newfile = "car.txt"
            new_file = open(newfile, "w")  # 用open函数建立一个新的文本文件
            new_file.close()
        carpath = os.path.abspath('.') + "/car.txt"
        os.system("notepad {}".format(carpath))

    # 视频播放
    @Slot()
    def start_capture(self, link=None):
        if link is None:
            link = self.defaultLink

        # if not self.isOpenLink(link):
        #     return False
        self.capture = cv2.VideoCapture(link)
        if self.capture.isOpened():
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.display_frame)
            self.timer.start(60)

    @Slot()
    def start_capture_1(self, link=None):
        if link is None:
            link = self.defaultLink

        # if not self.isOpenLink(link):
        #     return False
        try:
            self.capture1 = cv2.VideoCapture(link)
            # self.capture1.set(cv2.CAP_PROP_FFMPEG_TIMEOUT, 3000)
            if self.capture1.isOpened():
                self.timer1 = QTimer(self)
                self.timer1.timeout.connect(self.display_frame_1)
                self.timer1.start(60)
        except Exception as e:
            print(f'{str(e)}')

    @Slot()
    def start_capture_2(self, link=None):
        if link is None:
            link = self.defaultLink
        try:

            # if not self.isOpenLink(link):
            #     self.messageAlert(f'{str(link)}无法打开')
            #     return False
            self.capture2 = cv2.VideoCapture(link)
            # self.capture2.set(cv2.CAP_PROP_FFMPEG_TIMEOUT, 3000)
        except:
            pass
        if self.capture2.isOpened():
            self.timer2 = QTimer(self)
            self.timer2.timeout.connect(self.display_frame_2)
            self.timer2.start(60)

    @Slot()
    def start_capture_3(self, link=None):
        if link is None:
            link = self.defaultLink

        # if not self.isOpenLink(link):
        #     self.messageAlert(f'{str(link)}无法打开')
        #     return False
        try:
            self.capture3 = cv2.VideoCapture(link)
            # cv2.CAP_FFMPEG
            # self.capture3.set(cv2.CAP_PROP_FFMPEG_TIMEOUT, 3000)
        except:
            pass
        if self.capture3.isOpened():
            self.timer3 = QTimer(self)
            self.timer3.timeout.connect(self.display_frame_3)
            self.timer3.start(60)

    def display_frame(self):
        ret, frame = self.capture.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = frame.shape
            img = QImage(frame, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_3.size(),
                                                                           aspectMode=Qt.KeepAspectRatio)
            widgets.home_widget_3.setPixmap(QPixmap.fromImage(img))

    def display_frame_1(self):
        ret, frame = self.capture1.read()

        if ret:
            h, w, ch = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_4.size(),
                                                                                aspectMode=Qt.KeepAspectRatio)
            widgets.home_widget_4.setPixmap(QPixmap.fromImage(img))

    def display_frame_2(self):
        ret, frame = self.capture2.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = frame.shape

            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_5.size(),
                                                                                aspectMode=Qt.KeepAspectRatio)

            widgets.home_widget_6.setPixmap(QPixmap.fromImage(img))

    def display_frame_3(self):

        ret, frame = self.capture3.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            h, w, ch = frame.shape

            img = QImage(frame, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_6.size(),
                                                                           aspectMode=Qt.KeepAspectRatio)
            widgets.home_widget_5.setPixmap(QPixmap.fromImage(img))

    def isOpenLink(self, rtsp_url):

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 设置超时时间，以秒为单位
            host, port = rtsp_url.split("://")[1].split(":")  # 默认RTSP端口是554
            sock.connect((host, 554))
            print(f"RTSP服务器通畅：{rtsp_url}")
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            print(f"无法连接到RTSP服务器：{rtsp_url}")
            return False

    def getLinkNum(self):
        camera_access = []
        with open("./link.txt", 'r') as f:
            linkNum = 1
            for i in f.read().split("\n"):
                if i != "":
                    camera_access.append(i)
        return len(camera_access)

    @Slot()
    def Qmessage(self):
        QMessageBox(self, "提示", "视频合成完成！", QMessageBox.Ok)

    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    # ///////////////////////////////////////////////////////////////
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW WIDGETS PAGE
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW NEW PAGE
        if btnName == "btn_new":
            pass

            # widgets.stackedWidget.setCurrentWidget(widgets.new_page)  # SET PAGE
            # UIFunctions.resetStyle(self, btnName)  # RESET ANOTHERS BUTTONS SELECTED
            # btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))  # SELECT MENU

        if btnName == "playTopBtn":
            if self.isStart == 1:

                self.startbox = QMessageBox.question(self, '提醒', '确定要打开摄像头吗？', QMessageBox.StandardButton.Ok,
                                                     QMessageBox.StandardButton.No)
                print(self.startbox)

                if self.startbox == QMessageBox.StandardButton.Ok:
                    widgets.creditsLabel.setText("正在加载视频")

                    # self.out1 = None
                    # self.out2 = None
                    # self.out3 = None
                    # self.out4 = None

                    self.start_capture(self.link1)

                    self.start_capture_1(self.link2)

                    self.start_capture_2(self.link3)

                    self.start_capture_3(self.link4)
                    self.car = []
                    self.car.append(widgets.comboBox_4.currentText())
                    self.car.append(widgets.comboBox_5.currentText())
                    self.car.append(widgets.comboBox_6.currentText())
                    self.car.append(widgets.comboBox_7.currentText())
                    self.isStart = 0
                    icon2 = QIcon()
                    icon2.addFile(u":/icons/images/icons/pause.png", QSize(), QIcon.Normal, QIcon.Off)
                    widgets.playTopBtn.setIcon(icon2)
                    widgets.playTopBtn.setIconSize(QSize(20, 20))
                    widgets.creditsLabel.setText("加载完成，可以开启摄像头识别")
            else:
                self.endbox = QMessageBox.question(self, '提醒', '确定要关闭摄像头吗？', QMessageBox.StandardButton.Ok,
                                                   QMessageBox.StandardButton.No)

                if self.endbox == QMessageBox.StandardButton.Ok:

                    widgets.creditsLabel.setText("正在关闭视频")
                    try:
                        if self.timer != None:
                            self.timer.stop()
                        if self.capture != None:
                            self.capture.release()
                        self.capture = None

                        if self.timer1 != None:
                            self.timer1.stop()
                        if self.capture1 != None:
                            self.capture1.release()
                        # self.capture1.release()
                        self.capture1 = None
                        if self.timer2 != None:
                            self.timer2.stop()
                        if self.capture2 != None:
                            self.capture2.release()
                        # self.capture2.release()
                        self.capture2 = None
                        if self.timer3 != None:
                            self.timer3.stop()
                        if self.capture3 != None:
                            self.capture3.release()
                        # self.capture3.release()
                        self.capture3 = None
                    except Exception as e:
                        print(e)

                    widgets.home_widget_6.clear()
                    widgets.home_widget_5.clear()
                    widgets.home_widget_4.clear()
                    widgets.home_widget_3.clear()
                    self.isStart = 1
                    icon1 = QIcon()
                    icon1.addFile(u":/icons/images/icons/play.png", QSize(), QIcon.Normal, QIcon.Off)
                    widgets.playTopBtn.setIcon(icon1)
                    widgets.playTopBtn.setIconSize(QSize(20, 20))
                    widgets.creditsLabel.setText("视频已经关闭")

            # print("Save BTN clicked!")
        if btnName == "btn_exit":
            # print("0000")
            pass
        if btnName == "btn_save":
            # self.stop()
            pass
            # print("0000")

        # PRINT BTN NAME
        # print(f'Button "{btnName}" pressed!')

    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            pass
            # print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            pass
            # print('Mouse click: RIGHT CLICK')


if __name__ == "__main__":
    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    yolo = yolo_thread.YoloThread()
    window = MainWindow(yolo)

    try:
        app_exec = app.exec
    except AttributeError:
        app_exec = app.exec_
    sys.exit(app_exec())
