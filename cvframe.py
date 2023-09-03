import time

import cv2
import numpy as np
import torch
from torch import device


def nms(pred, conf_thres, iou_thres):
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = getInter(max_conf_box, current_box)
                iou = getIou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0] - box1[2] / 2, box1[1] - box1[3] / 2, \
                                         box1[0] + box1[2] / 2, box1[1] + box1[3] / 2
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0] - box2[2] / 2, box2[1] - box1[3] / 2, \
                                         box2[0] + box2[2] / 2, box2[1] + box2[3] / 2
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def draw(img, xscale, yscale, pred):
    img_ = img.copy()
    if len(pred):
        for detect in pred:
            detect = [int((detect[0] - detect[2] / 2) * xscale), int((detect[1] - detect[3] / 2) * yscale),
                      int((detect[0] + detect[2] / 2) * xscale), int((detect[1] + detect[3] / 2) * yscale)]
            img_ = cv2.rectangle(img, (detect[0], detect[1]), (detect[2], detect[3]), (0, 255, 0), 1)
    return img_


def process_frame(img_bgr, ort_session):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''
    input_shape = ort_session.get_inputs()[0].shape
    input_height, input_width = [640, 640]
    # print(type(input_width))
    # print(input_width,input_height)
    # X 方向 图像缩放比例
    x_ratio = float(img_bgr.shape[1]) / float(input_width)
    # Y 方向 图像缩放比例
    y_ratio = float(img_bgr.shape[0]) / float(input_height)
    kpts_shape = [3, 3]
    # 框（rectangle）可视化配置
    bbox_color = (150, 0, 0)  # 框的 BGR 颜色
    bbox_thickness = 2  # 框的线宽

    # 框类别文字
    bbox_labelstr = {
        'font_size': 1,  # 字体大小
        'font_thickness': 2,  # 字体粗细
        'offset_x': 0,  # X 方向，文字偏移距离，向右为正
        'offset_y': -10,  # Y 方向，文字偏移距离，向下为正
    }
    kpt_color_map = {
        0: {'name': 'angle_30', 'color': [255, 0, 0], 'radius': 6},  # 30度角点
        1: {'name': 'angle_60', 'color': [0, 255, 0], 'radius': 6},  # 60度角点
        2: {'name': 'angle_90', 'color': [0, 0, 255], 'radius': 6},  # 90度角点
    }

    # 点类别文字
    kpt_labelstr = {
        'font_size': 1.5,  # 字体大小
        'font_thickness': 3,  # 字体粗细
        'offset_x': 10,  # X 方向，文字偏移距离，向右为正
        'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
    }

    # 骨架连接 BGR 配色
    skeleton_map = [
        {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [196, 75, 255], 'thickness': 2},  # 30度角点-60度角点
        {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [180, 187, 28], 'thickness': 2},  # 30度角点-90度角点
        {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [47, 255, 173], 'thickness': 2},  # 60度角点-90度角点
    ]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    # 记录该帧开始处理的时间
    start_time = time.time()

    # 预处理-缩放图像尺寸
    img_bgr_640 = cv2.resize(img_bgr, [input_height, input_width])
    img_rgb_640 = img_bgr_640[:, :, ::-1]
    # 预处理-归一化
    input_tensor = img_rgb_640 / 255
    # 预处理-构造输入 Tensor
    input_tensor = np.expand_dims(input_tensor, axis=0)  # 加 batch 维度
    input_tensor = input_tensor.transpose((0, 3, 1, 2))  # N, C, H, W
    input_tensor = np.ascontiguousarray(input_tensor)  # 将内存不连续存储的数组，转换为内存连续存储的数组，使得内存访问速度更快
    input_tensor = torch.from_numpy(input_tensor).to(device).float()  # 转 Pytorch Tensor
    # input_tensor = input_tensor.half() # 是否开启半精度，即 uint8 转 fp16，默认转 fp32

    # ONNX Runtime 推理预测
    input_name = ort_session.get_inputs()[0].name
    ort_output = ort_session.run(None, {input_name: input_tensor.numpy()})[0]
    pred = np.squeeze(ort_output)
    pred = np.transpose(pred, (1, 0))
    pred_class = pred[..., 4:]
    # print("pred_class:", pred_class)
    # print(pred_class.size)
    if len(pred_class) > 0:
        try:
            pred_conf = np.argmax(pred_class, axis=-1)
        except ValueError:
            # print(ValueError)
            return img_bgr

    else:
        return img_bgr
    # print("pred_conf:",pred_conf)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    result = nms(pred, 0.3, 0.4)
    print("result:",result)
    ret_img = draw(img_bgr, x_ratio, y_ratio, result)


    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)
    # print(FPS)
    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  {:.2f}'.format(FPS)  # 写在画面上的字符串
    # ret_img = cv2.putText(ret_img, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)
    print(FPS_string)
    return ret_img
