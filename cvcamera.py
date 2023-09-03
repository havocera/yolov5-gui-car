import time

import cv2
import numpy as np
import onnxruntime

from cvframe import process_frame


class CVCamera:
    def __init__(self):
        try:
            with open("./link.txt", 'r') as f:
                self.link = f.read().split("\n")

        except:
            pass

    def start_cv(self):
        for i in self.link:
            self.getlinktoframe(i)

    def getlinktoframe(self, link):
        try:
            cap = cv2.VideoCapture(link)
        except:
            return
        while True:
            ret, frame = cap.read()

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame = process_frame(frame, self.onnx_model)

                cv2.imshow(frame)

import ffmpeg
probe = ffmpeg.probe("./video.mp4")
video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
width = int(video_stream['width'])
height = int(video_stream['height'])
# out = (
#     ffmpeg
#         .input("./video.mp4", rtsp_transport='tcp')
#         .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=25)
#         .run_async(pipe_stdout=True)
# )
out = (
    ffmpeg
        .input("./video.mp4")
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', loglevel="quiet", r=25)
        .run_async(pipe_stdout=True)
)
cnt_empty = 0
session = onnxruntime.InferenceSession('bestv5s.onnx',providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# from ultralytics import YOLO

# model = YOLO("./ptmodel/best.pt")
# print(1)

while True:
    in_bytes = out.stdout.read(height * width * 3)
    if not in_bytes:
        cnt_empty += 1
        if cnt_empty > 10:
            break
    cnt_empty = 0
    # print(in_bytes)
    # start_time = time.time()
    frame = np.frombuffer(in_bytes, dtype=np.uint8).reshape(height,width,3)
    # results = model(frame)
    # end_time = time.time()
    # print(1 / (end_time - start_time))
    frame = process_frame(frame,session)
    # frame = np.reshape(frame,(height,width,3))
    # to process frame
    # cv2.imshow('test', frame)
    if 0xFF == ord('q'):
        break


if __name__ == '__main__':
    pass
    # cvshow = CVCamera()
    # cvshow.start_cv()