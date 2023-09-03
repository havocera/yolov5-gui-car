"""
@ capture multiple cameras images to different folder
"""

import cv2
import numpy as np
import time
import os
import threading
from time import ctime, sleep
import queue

import onnxruntime
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidGraph

import v5frame

abspath = os.path.abspath(os.path.dirname(__file__))


class threadCameraRSTP(threading.Thread):
    """Hikvision camera
    @user   User name.
    @passwd User password.
    @ip     Camera ip name.
    @queue  Output queue.
    """

    def __init__(self, url, model):
        threading.Thread.__init__(self)
        self.url = url
        self.model = model
        self.q = queue




class threadCameraUSB(threading.Thread):
    """usb camera
    @access   Usb descriptor.
    @queue    Output queue.
    """

    def __init__(self, access, queue):
        threading.Thread.__init__(self)
        self.access = access
        self.q = queue

    def run(self):
        cap = cv2.VideoCapture(self.access)
        if cap.isOpened():
            print('camera usb ' + str(self.access) + " connected.")

        while True:
            ret, img = cap.read()
            if ret:
                self.q.put(img)
                self.q.get() if self.q.qsize() > 2 else time.sleep(0.01)


def image_save(queueImage, queueCmd, dstDir, startNum, identification, display=True):
    """save image
    @queueImage   Image input queue.
    @queueCmd     Command input queue.
    @dstDir       Folder storing images.
    @startNum     Images number.
    @identification     Image show name.
    @display       Whether display image.
    """
    count = startNum
    if display:
        cv2.namedWindow(identification, 0)
    img = None

    while (True):
        if queueCmd.qsize() > 0:
            cmd = queueCmd.get()
            if cmd == 's':
                while True:
                    if queueImage.qsize() > 0:
                        img = queueImage.get()
                        cv2.imwrite(dstDir + '/' + ('%s' % count) + ".png", img)
                        print('n' + identification + ' save img ' + str(count) + ' OK')
                        count += 1
                        break

        if queueImage.qsize() > 1:
            img = queueImage.get()
            if display:
                cv2.imshow(identification, img)

        cv2.waitKey(1)





queueCmds = []
thread_ids = []


def captureMutipleCamera(onnx_session,camera_access=[]):
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

    ## 1.start camera threads and save threads
    for camera in camera_access:
        identification = None
        cameraThread = None
        dstDir = None
        queueImage = queue.Queue(maxsize=4)
        queueCmd = queue.Queue(maxsize=4)

        cameraThread = threadCameraRSTP(camera,onnx_session)
        # camera thread
        thread_ids.append(cameraThread)

        # save image thread
        # thread_ids.append(threading.Thread(target=image_save, args=(
        #     queueImage, queueCmd, dstDir, start_Num, identification, display)))

        # cmd input queue
        queueCmds.append(queueCmd)

    for thread in thread_ids:
        thread.daemon = True
        thread.start()


def stop():
    for thread in thread_ids:
        thread.daemon = False
        thread.stop()


if __name__ == '__main__':
    """example to capture camera
    """
    camera_access = [
        "",
        "",
    ]

    captureMutipleCamera(camera_access)
