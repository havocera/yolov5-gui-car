# ///////////////////////////////////////////////////////////////
#
# ///////////////////////////////////////////////////////////////
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
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph

import cv2
import sys
import sys

# 里面替换为自己项目目录下的文件路径
sys.path.insert(0, os.path.abspath('.'))
import onnxruntime

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
        QTimer.singleShot(800, loop.quit)
        loop.exec_()
        QApplication.processEvents()


class MainWindow(QMainWindow):
    resSignal = Signal(str)

    def __init__(self):
        QMainWindow.__init__(self)

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
        self.fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')
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
        '''
         onnx配置
        '''
        # import onnxruntime

        # print(onnxruntime.get_device())
        # self.onnx_model = self.load_onnx_model("./ptmodel/best.onnx")
        self.onnx_path = 'bestv5s.onnx'
        self.onnx_session = self.load_onnx_model()
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

        # self.model.isClass.connect(self.isclass)
        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)

        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)
        self.input_name = self.get_input_name()  # ['images']
        self.output_name = self.get_output_name()  # ['output0']
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

        self.resSignal.connect(self.isclass)
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
        self.link2 = self.defaultLink
        self.link3 = self.defaultLink
        self.link4 = self.defaultLink
        self.out1 = None
        self.out2 = None
        self.out3 = None
        self.out4 = None
        widgets.linkTable.setColumnCount(2)
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
        with open("log.txt","a") as f:
            f.write(text+"\n")

        # widgets.creditsLabel.setText(text)

    def start_yolo(self):
        camera_access = []
        with open("./link.txt", 'r') as f:
            linkNum = 1
            for i in f.read().split("\n"):
                if i != "":
                    camera_access.append(i)
        self.captureMutipleCamera(camera_access)

    def isclass(self, data):
        car = []
        car.append(widgets.comboBox_4.currentText())
        car.append(widgets.comboBox_5.currentText())
        car.append(widgets.comboBox_6.currentText())
        car.append(widgets.comboBox_7.currentText())
        print(car)
        data = json.loads(data)
        print(data)
        try:
            frame_num = car.index(data["class"])
        except ValueError:
            print(ValueError)
            return 0
        if frame_num == 1:
            if self.link1 != data["link"]:
                self.link1 = data["link"]
                if self.timer == None:
                    self.start_capture(data["link"])
                else:
                    self.out1.release()
                    self.capture = None
                    self.timer.close()
                    self.start_capture(data["link"])

        elif frame_num == 2:
            if self.link2 != data["link"]:
                self.link2 = data["link"]
                if self.timer1 == None:
                    self.start_capture_1(data["link"])
                else:
                    self.out2.release()
                    self.capture1 = None
                    self.timer1.close()
                    self.start_capture_1(data["link"])
        elif frame_num == 3:
            if self.link3 != data["link"]:
                self.link3 = data["link"]
                if self.timer2 == None:
                    self.start_capture_2(data["link"])
                else:
                    self.out3.release()
                    self.capture2 = None
                    self.timer2.close()
                    self.start_capture_2(data["link"])
        elif frame_num == 4:
            if self.link4 != data["link"]:
                self.link4 = data["link"]
                if self.timer3 == None:
                    self.start_capture_3(data["link"])
                else:
                    self.out4.release()
                    self.capture3 = None
                    self.timer3.close()
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

    def load_onnx_model(self):
        try:
            session = onnxruntime.InferenceSession("bestv5s.onnx",
                                                   providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider',
                                                              'CPUExecutionProvider'])

        except (InvalidGraph, TypeError, RuntimeError) as e:
            # It is possible for there to be a mismatch between the onnxruntime and the
            # version of the onnx model format.
            print(e)
            raise e
        return session

    def show_result(self, statistic_dic):
        print(statistic_dic)

    # 视频播放
    @Slot()
    def start_capture(self, link=None):
        if link is None:
            link = self.defaultLink
        if not os.path.exists(self.base_path + "1/" + self.strpath + "/"):
            os.makedirs(self.base_path + "1/" + self.strpath + "/")
        if not self.isOpenLink(link):
            return False
        self.capture = cv2.VideoCapture(link)
        if self.capture.isOpened():
            filename = self.base_path + "1/" + self.strpath + "/1" + "_" + datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + ".mp4"

            self.out1 = cv2.VideoWriter(filename, self.fourcc, 30, (1280, 720))
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.display_frame)
            self.timer.start(60)

    @Slot()
    def start_capture_1(self, link=None):
        if link is None:
            link = self.defaultLink
        if not os.path.exists(self.base_path + "2/" + self.strpath + "/"):
            os.makedirs(self.base_path + "2/" + self.strpath + "/")
        if not self.isOpenLink(link):
            self.messageAlert(f'{str(link)}无法打开')
            return False
        try:
            self.capture1 = cv2.VideoCapture(link, cv2.CAP_FFMPEG)
            self.capture1.set(cv2.CAP_PROP_FFMPEG_TIMEOUT, 3000)
            if self.capture1.isOpened():
                filename = self.base_path + "2/" + self.strpath + "/2" + "_" + datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S") + ".mp4"
                self.out2 = cv2.VideoWriter(filename, self.fourcc, 30, (1280, 720))
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
            if not os.path.exists(self.base_path + "3/" + self.strpath + "/"):
                os.makedirs(self.base_path + "3/" + self.strpath + "/")
            if not self.isOpenLink(link):
                self.messageAlert(f'{str(link)}无法打开')
                return False
            self.capture2 = cv2.VideoCapture(link, cv2.CAP_FFMPEG)
            self.capture2.set(cv2.CAP_PROP_FFMPEG_TIMEOUT, 3000)
        except:
            pass
        if self.capture2.isOpened():
            filename = self.base_path + "3/" + self.strpath + "/3" + "_" + datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + ".mp4"

            self.out3 = cv2.VideoWriter(filename, self.fourcc, 30, (1280, 720))
            self.timer2 = QTimer(self)
            self.timer2.timeout.connect(self.display_frame_2)
            self.timer2.start(60)

    @Slot()
    def start_capture_3(self, link=None):
        if link is None:
            link = self.defaultLink
        if not os.path.exists(self.base_path + "4/" + self.strpath + "/"):
            os.makedirs(self.base_path + "4/" + self.strpath + "/")
        if not self.isOpenLink(link):
            self.messageAlert(f'{str(link)}无法打开')
            return False
        try:
            self.capture3 = cv2.VideoCapture(link, cv2.CAP_FFMPEG)
            self.capture3.set(cv2.CAP_PROP_FFMPEG_TIMEOUT, 3000)
        except:
            pass
        if self.capture3.isOpened():
            filename = self.base_path + "4/" + self.strpath + "/4" + "_" + datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + ".mp4"

            self.out4 = cv2.VideoWriter(filename, self.fourcc, 30, (1280, 720))
            self.timer3 = QTimer(self)
            self.timer3.timeout.connect(self.display_frame_3)
            self.timer3.start(60)

    def display_frame(self):
        ret, frame = self.capture.read()
        print("video---1")
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out1.write(frame)
            h, w, ch = frame.shape
            img = QImage(frame, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_3.size(),
                                                                           aspectMode=Qt.KeepAspectRatio)
            widgets.home_widget_3.setPixmap(QPixmap.fromImage(img))

    def display_frame_1(self):
        ret, frame = self.capture1.read()
        print("video---2")
        if ret:
            h, w, ch = frame.shape
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.out2.write(frame)

            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_4.size(),
                                                                                aspectMode=Qt.KeepAspectRatio)
            widgets.home_widget_4.setPixmap(QPixmap.fromImage(img))

    def display_frame_2(self):
        ret, frame = self.capture2.read()
        print("video---3")
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out3.write(frame)
            h, w, ch = frame.shape

            img = QImage(frame.data, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_5.size(),
                                                                                aspectMode=Qt.KeepAspectRatio)

            widgets.home_widget_6.setPixmap(QPixmap.fromImage(img))

    def display_frame_3(self):

        ret, frame = self.capture3.read()
        print("video---4")
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out4.write(frame)
            h, w, ch = frame.shape

            img = QImage(frame, w, h, ch * w, QImage.Format_RGB888).scaled(widgets.home_widget_6.size(),
                                                                           aspectMode=Qt.KeepAspectRatio)

            widgets.home_widget_5.setPixmap(QPixmap.fromImage(img))

    def threadCameraRSTP(self, url):
        access = url

        cap = cv2.VideoCapture(access)
        if cap.isOpened():
            pass

        while True:
            if not self.isStopyolo:
                return
            ret, img = cap.read()

            if ret:
                self.run(img, url)
            else:
                cap.release()
                break

    def captureMutipleCamera(self, camera_access=[]):
        """save image
        @camera_access  All paremeters to capture and save the image, list, the format lile,
                        [
                            ["HIKVISION", "admin", "aaron20127", "192.168.0.111", 'D:/data/image'],
                            ["USB", 0, 'D:/data/image']
                        ]
        @start_Num     Image show name.
        @display       Whether display image.
        """

        # there must import queue, I don't know why
        import queue
        self.thread_ids.clear()
        ## 1.start camera threads and save threads
        for camera in camera_access:
            if self.isOpenLink(camera):

                identification = None
                cameraThread = None
                dstDir = None
                queueImage = queue.Queue(maxsize=4)
                queueCmd = queue.Queue(maxsize=4)
                cameraThread = threading.Thread(target=self.threadCameraRSTP, args=(camera,))

                # camera thread
                self.thread_ids.append(cameraThread)
            else:
                self.noopen.append(camera)

        for thread in self.thread_ids:
            # thread.daemon = True
            thread.start()

    def stop(self):
        for thread in self.thread_ids:
            thread.join()
            del thread

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

    def messageAlert(self, content):
        msg = QMessageBox()
        # 设置非模态
        msg.setWindowModality(Qt.NonModal)
        # self.msg.setFixedSize(400,250)
        msg.setStyleSheet("QLabel{"
                          "min-width: 200px;"
                          "min-height: 100px; "
                          "}")
        # 设置弹窗标题和内容
        msg.setWindowTitle('警告')
        msg.setText(content)
        # 设置弹窗的按钮为OK，StandardButtons采用位标记，可以用与运算添加其他想要的按钮
        msg.setStandardButtons(QMessageBox.Ok)
        # 显示窗口
        msg.show()

    def getLinkNum(self):
        camera_access = []
        with open("./link.txt", 'r') as f:
            linkNum = 1
            for i in f.read().split("\n"):
                if i != "":
                    camera_access.append(i)
        return len(camera_access)
    def createVideo(self):
        import cv2
        import os
        import random
        import glob
        for i in range(1, 5, 1):
            path = os.path.abspath('.') + "/video/{}/".format(i)

            filename = path + "/" + datetime.datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S") + "汇总" + ".mp4"

            VideoWriter = cv2.VideoWriter(filename, self.fourcc, 30, (1280, 720))
            # VideoWriter = cv2.VideoWriter("merge .avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 24, (600, 480))
            mp4list = glob.glob(os.path.join(path + self.strpath + "/", "*.mp4"))
            # print(mp4list)
            if len(mp4list) == 0:
                continue
            for mp4file in mp4list:
                capture = cv2.VideoCapture(mp4file)
                # print(mp4file)
                while True:

                    ret, prev = capture.read()
                    # print(ret)
                    if ret:
                        VideoWriter.write(prev)
                    else:
                        break
            VideoWriter.release()

        QMessageBox.information(self, "提示",
                                "视频已合并完成!",
                                QMessageBox.StandardButton.Ok)
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
                self.link1 = self.defaultLink
                self.link2 = self.defaultLink
                self.link3 = self.defaultLink
                self.link4 = self.defaultLink
                self.out1 = None
                self.out2 = None
                self.out3 = None
                self.out4 = None
                if self.isOpenLink(self.link1):
                    self.start_capture(self.link1)
                if self.isOpenLink(self.link2):
                    self.start_capture_1(self.link2)
                if self.isOpenLink(self.link3):
                    self.start_capture_2(self.link3)
                if self.isOpenLink(self.link4):
                    self.start_capture_3(self.link4)

                self.strpath = datetime.datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S")
                self.noopen = []
                self.start_yolo()
                if len(self.noopen) != self.getLinkNum():
                    pass
                else:
                    print("请打开摄像头")
                self.isStart = 0
                icon2 = QIcon()
                icon2.addFile(u":/icons/images/icons/pause.png", QSize(), QIcon.Normal, QIcon.Off)
                widgets.playTopBtn.setIcon(icon2)
                widgets.playTopBtn.setIconSize(QSize(20, 20))

            else:
                self.isStopyolo = False
                self.stop()
                if self.timer != None:
                    self.timer.stop()
                if self.capture != None:
                    self.out1.release()
                    self.capture.release()
                self.capture = None

                if self.timer1 != None:
                    self.timer1.stop()
                if self.capture1 != None:
                    self.out2.release()
                    self.capture1.release()
                # self.capture1.release()
                self.capture1 = None
                if self.timer2 != None:
                    self.timer2.stop()
                if self.capture2 != None:
                    self.out3.release()
                    self.capture2.release()
                # self.capture2.release()
                self.capture2 = None
                if self.timer3 != None:
                    self.timer3.stop()
                if self.capture3 != None:
                    self.out4.release()
                    self.capture3.release()
                # self.capture3.release()
                self.capture3 = None

                widgets.home_widget_6.clear()
                widgets.home_widget_5.clear()
                widgets.home_widget_4.clear()
                widgets.home_widget_3.clear()
                self.isStart = 1
                icon1 = QIcon()
                icon1.addFile(u":/icons/images/icons/play.png", QSize(), QIcon.Normal, QIcon.Off)
                widgets.playTopBtn.setIcon(icon1)
                widgets.playTopBtn.setIconSize(QSize(20, 20))

                import cv2
                import os
                import random
                import glob
                video_thread = threading.Thread(target=self.createVideo)
                video_thread.start()
                self.isStopyolo = True


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
            '''
            =====================yolo=========================
            '''

    def get_input_name(self):
        """获取输入节点名称"""
        input_name = []
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)

        return input_name

    def get_output_name(self):
        """获取输出节点名称"""
        output_name = []
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)

        return output_name

    def get_input_feed(self, image_numpy):
        """获取输入numpy"""
        input_feed = {}
        for name in self.input_name:
            input_feed[name] = image_numpy

        return input_feed

    def inference(self, img_path):
        """ 1.cv2读取图像并resize
        2.图像转BGR2RGB和HWC2CHW(因为yolov5的onnx模型输入为 RGB：1 × 3 × 640 × 640)
        3.图像归一化
        4.图像增加维度
        5.onnx_session 推理 """
        # img = cv2.imread(img_path)
        or_img = cv2.resize(img_path, (640, 640))  # resize后的原图 (640, 640, 3)
        img = or_img[:, :, ::-1].transpose(2, 0, 1)  # BGR2RGB和HWC2CHW
        img = img.astype(dtype=np.float32)  # onnx模型的类型是type: float32[ , , , ]
        img /= 255.0
        img = np.expand_dims(img, axis=0)  # [3, 640, 640]扩展为[1, 3, 640, 640]
        # img尺寸(1, 3, 640, 640)
        input_feed = self.get_input_feed(img)  # dict:{ input_name: input_value }
        pred = self.onnx_session.run(None, input_feed)[0]  # <class 'numpy.ndarray'>(1, 25200, 9)

        return pred, or_img

    # dets:  array [x,6] 6个值分别为x1,y1,x2,y2,score,class
    # thresh: 阈值
    def nms(self, dets, thresh):
        # dets:x1 y1 x2 y2 score class
        # x[:,n]就是取所有集合的第n个数据
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        # -------------------------------------------------------
        #   计算框的面积
        #	置信度从大到小排序
        # -------------------------------------------------------
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        # print(scores)
        keep = []
        index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
        # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

        while index.size > 0:
            i = index[0]
            keep.append(i)
            # -------------------------------------------------------
            #   计算相交面积
            #	1.相交
            #	2.不相交
            # -------------------------------------------------------
            x11 = np.maximum(x1[i], x1[index[1:]])
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h
            # -------------------------------------------------------
            #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
            #	IOU小于thresh的框保留下来
            # -------------------------------------------------------
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep

    def xywh2xyxy(self, x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def filter_box(self, org_box, conf_thres, iou_thres):  # 过滤掉无用的框
        # -------------------------------------------------------
        #   删除为1的维度
        #	删除置信度小于conf_thres的BOX
        # -------------------------------------------------------
        org_box = np.squeeze(org_box)  # 删除数组形状中单维度条目(shape中为1的维度)
        # (25200, 9)
        # […,4]：代表了取最里边一层的所有第4号元素，…代表了对:,:,:,等所有的的省略。此处生成：25200个第四号元素组成的数组
        conf = org_box[..., 4] > conf_thres  # 0 1 2 3 4 4是置信度，只要置信度 > conf_thres 的
        box = org_box[conf == True]  # 根据objectness score生成(n, 9)，只留下符合要求的框
        # print('box:符合要求的框')
        # print(box.shape)

        # -------------------------------------------------------
        #   通过argmax获取置信度最大的类别
        # -------------------------------------------------------
        cls_cinf = box[..., 5:]  # 左闭右开（5 6 7 8），就只剩下了每个grid cell中各类别的概率
        cls = []
        for i in range(len(cls_cinf)):
            cls.append(int(np.argmax(cls_cinf[i])))  # 剩下的objecctness score比较大的grid cell，分别对应的预测类别列表
        all_cls = list(set(cls))  # 去重，找出图中都有哪些类别
        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        # -------------------------------------------------------
        #   分别对每个类别进行过滤
        #   1.将第6列元素替换为类别下标
        #	2.xywh2xyxy 坐标转换
        #	3.经过非极大抑制后输出的BOX下标
        #	4.利用下标取出非极大抑制后的BOX
        # -------------------------------------------------------
        output = []
        for i in range(len(all_cls)):
            curr_cls = all_cls[i]
            curr_cls_box = []
            curr_out_box = []

            for j in range(len(cls)):
                if cls[j] == curr_cls:
                    box[j][5] = curr_cls
                    curr_cls_box.append(box[j][:6])  # 左闭右开，0 1 2 3 4 5

            curr_cls_box = np.array(curr_cls_box)  # 0 1 2 3 4 5 分别是 x y w h score class
            # curr_cls_box_old = np.copy(curr_cls_box)
            curr_cls_box = self.xywh2xyxy(curr_cls_box)  # 0 1 2 3 4 5 分别是 x1 y1 x2 y2 score class
            curr_out_box = self.nms(curr_cls_box, iou_thres)  # 获得nms后，剩下的类别在curr_cls_box中的下标

            for k in curr_out_box:
                output.append(curr_cls_box[k])
        output = np.array(output)
        return output

    def draw(self, image, box_data, url):
        # -------------------------------------------------------
        #	取整，方便画框
        # -------------------------------------------------------

        boxes = box_data[..., :4].astype(np.int32)  # x1 x2 y1 y2
        scores = box_data[..., 4]
        classes = box_data[..., 5].astype(np.int32)
        # print(classes)

        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            isclassdata = {"link": url, "class": self.CLASSES[cl]}
            self.resSignal.emit(json.dumps(isclassdata))

            # print('class: {}, score: {}'.format(CLASSES[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
                        (top, left),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)
        return image

    def run(self, frame, url):

        output, or_img = self.inference(frame)
        outbox = self.filter_box(output, 0.5, 0.5)
        if len(outbox) == 0:
            pass
            # print('没有发现物体')
            # sys.exit(0)
        else:

            or_img = self.draw(or_img, outbox, url)
        return or_img


'''
===================yolo

'''

if __name__ == "__main__":

    if hasattr(QtCore.Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    if hasattr(QtCore.Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    try:
        app_exec = app.exec
    except AttributeError:
        app_exec = app.exec_
    sys.exit(app_exec())
