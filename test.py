# import onnx
# import onnxruntime
#
# # 读取 ONNX 模型
# onnx_model = onnx.load('./ptmodel/best.onnx')
#
# # 检查模型格式是否正确
# onnx.checker.check_model(onnx_model)
#
# print('无报错，onnx模型载入成功')
# print(onnx.helper.printable_graph(onnx_model.graph))
# session = onnxruntime.InferenceSession('./ptmodel/best.onnx',providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
# input_name = session.get_inputs()
# label_name = session.get_outputs()
# print(input_name,label_name)
# for i in label_name:
#     print(i.shape[2:])
#     print(i)
# # session.run()
import time

import cv2

# Load a model
# model = YOLO('yolov8n.pt')  # load an official model
# model = YOLO('bestv5s.onnx')  # load a custom trained
#
# # Export the model
# cap = cv2.VideoCapture("./video.mp4")
# while True:
#     ret,frame = cap.read()
#     start_time = time.time()
#     if ret:
#         res = model(frame)
#         print(res)
#     end_time = time.time()
#     print(1/(end_time-start_time))
#     # cv2.imshow(res["orig_img"])
for (i, index) in range(1, 9):
    print(i, index)
