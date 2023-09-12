import datetime
import json
import multiprocessing
import os
import socket
import threading

import cv2
import numpy as np
import onnxruntime
from PySide6.QtCore import QObject, Signal

from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph


class VideoRecorder():

    def __init__(self, num):
        self.capture = None
        self.out1 = None
        # self.output_path = output_path
        self.last_url = ""
        self.base_path = os.path.abspath('.') + "/video/"
        self.is_recording = threading.Event()
        self.strpath = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.num = str(num)
        self.log(self.num)

    def run(self, url):

        if self.last_url != url:

            self.log(url)
            self.last_url = url
            if self.out1:
                self.out1.release()
                self.capture.release()
                self.stop()

            self.start_recording(url)

    def log(self, log):
        with open("yolo_log.txt", "a", encoding="utf-8") as f:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d,%H:%M:%S")

            f.write("-----------" + formatted_datetime + "-----------\n" + str(log) + "\n")

    def stop(self):
        self.out1.release()
        self.capture.release()

        self.is_recording.clear()

    def start_recording(self, url):
        self.is_recording.set()
        self.log("录像子线程")
        rec_video = threading.Thread(target=self.record_video, args=(url, self.num))
        rec_video.start()


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

    def record_video(self, link, num):
        if not os.path.exists(self.base_path + num + "/" + self.strpath + "/"):
            os.makedirs(self.base_path + num + "/" + self.strpath + "/")

        self.capture = cv2.VideoCapture(link)

        filename = self.base_path + num + "/" + self.strpath + "/" + num + "_" + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S") + ".mp4"
        self.out1 = cv2.VideoWriter(filename, self.fourcc, 30, (1280, 720))
        self.log(f"开始录像，通道{num},连接：{link}")
        while self.is_recording:
            ret, frame = self.capture.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.out1.write(frame)


class YoloThread(QObject):
    yoloSignl = Signal(str)
    yoloStatusSignl = Signal(str)

    def __init__(self):
        super().__init__()
        self.recorders = {
            "stream1": VideoRecorder(1),
            "stream2": VideoRecorder(2),
            "stream3": VideoRecorder(3),
            "stream4": VideoRecorder(4),
        }
        with open("car.txt", "r") as f:
            cardata = f.read().split("\n")
            self.CLASSES = cardata
        self.onnx_session = self.load_onnx_model()
        self.isStopyolo = True
        self.input_name = self.get_input_name()  # ['images']
        self.output_name = self.get_output_name()  # ['output0']
        self.thread_ids = []
        self.noopen = []
        self.lastUrl1 = ''
        self.lastUrl2 = ''
        self.lastUrl3 = ''
        self.lastUrl4 = ''
        self.strpath = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self._frontUrl = ""
        self._frontClass = ""

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

    def threadCameraRSTP(self, url):
        access = url

        cap = cv2.VideoCapture(access)
        if cap.isOpened():
            pass

        while self.isStopyolo:

            ret, img = cap.read()

            if ret:
                self.run(img, url)
            else:
                cap.release()
                break

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
            num = cl + 1
            self.recorders["stream" + str(num)].run(url)
            message = json.dumps(isclassdata)

            self.yoloSignl.emit(message)
            self.log(message)

            # print('class: {}, score: {}'.format(CLASSES[cl], score))
            # print('box coordinate left,top,right,down: [{}, {}, {}, {}]'.format(top, left, right, bottom))

            # cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            # cv2.putText(image, '{0} {1:.2f}'.format(self.CLASSES[cl], score),
            #             (top, left),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, (0, 0, 255), 2)
        # return image

    def log(self, log):
        with open("yolo_log.txt", "a", encoding="utf-8") as f:
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d,%H:%M:%S")

            f.write("-----------" + formatted_datetime + "-----------\n" + log + "\n")

    def isOpenLink(self, rtsp_url):

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)  # 设置超时时间，以秒为单位
            host, port = rtsp_url.split("://")[1].split(":")  # 默认RTSP端口是554
            sock.connect((host, 554))
            self.log(f"RTSP服务器通畅：{rtsp_url}")
            print(f"RTSP服务器通畅：{rtsp_url}")
            sock.close()
            return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            print(f"无法连接到RTSP服务器：{rtsp_url}")
            return False

    def captureMutipleCamera(self):
        """save image
        @camera_access  All paremeters to capture and save the image, list, the format lile,
                        [
                            ["HIKVISION", "admin", "aaron20127", "192.168.0.111", 'D:/data/image'],
                            ["USB", 0, 'D:/data/image']
                        ]
        @start_Num     Image show name.
        @display       Whether display image.
        """
        self.strpath = datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S")

        camera_access = []
        self.yoloStatusSignl.emit(json.dumps({"status": 1, "message": "正在获取视频链接"}))
        with open("./link.txt", 'r') as f:
            linkNum = 1
            for i in f.read().split("\n"):
                if i != "":
                    camera_access.append(i)

        # there must import queue, I don't know why
        import queue
        self.thread_ids.clear()
        ## 1.start camera threads and save threads
        # num_processes = 20
        # pool = multiprocessing.Pool(processes=num_processes)
        self.log("正在开启录像")
        self.yoloStatusSignl.emit(json.dumps({"status": 1, "message": "正在开启录像"}))
        for i in range(1, 5):
            self.recorders["stream" + str(i)].strpath = self.strpath
            self.recorders["stream" + str(i)].run(camera_access[0])
        self.log("正在开启多线程")
        self.yoloStatusSignl.emit(json.dumps({"status": 1, "message": "正在开启多线程"}))
        for camera in camera_access:
            # if self.isOpenLink(camera):
            if True:
                identification = None
                cameraThread = None
                dstDir = None
                queueImage = queue.Queue(maxsize=4)
                queueCmd = queue.Queue(maxsize=4)
                cameraThread = threading.Thread(target=self.threadCameraRSTP, args=(camera,))

                # pool.apply_async(self.threadCameraRSTP, args=(camera,))
                # camera thread
                self.thread_ids.append(cameraThread)
            else:
                self.noopen.append(camera)
        for thread in self.thread_ids:
            thread.start()
        self.log("线程已启用")
        self.yoloStatusSignl.emit(json.dumps({"status": 1, "message": "线程已启用"}))

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
        self.log("视频完事了")
        print("视频完事了")

def start_keyboard_listener():
    """
    开始键盘监听的回调函数
    """
    print("Ctrl+Shift+A pressed")

    yolo.captureMutipleCamera()
    # do something


def stop_keyboard_listener():
    """
    停止键盘监听的回调函数
    """
    print("Ctrl+Shift+Q pressed")
    yolo.isStopyolo = False
    thread_count = 0
    for i in yolo.thread_ids:
        i.join()
        if i.is_alive():
            thread_count += 1

    for num in range(1, 5):
        yolo.recorders["stream" + str(num)].stop()
    yolo.createVideo()

    # do something

if __name__ == '__main__':
    import keyboard
    yolo = YoloThread()
    keyboard.add_hotkey('Ctrl+Shift+A', start_keyboard_listener)
    keyboard.add_hotkey('Ctrl+Shift+Q', stop_keyboard_listener)
    try:
        keyboard.wait('ctrl+c')
    except KeyboardInterrupt:
        print(KeyboardInterrupt)