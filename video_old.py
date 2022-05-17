import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
from xml.dom.minidom import parse, parseString
import xml.etree.ElementTree as ET
import matplotlib.patches as patches

import torch
from torchsummary import summary
import torchvision
from torchvision import transforms

import os
import pathlib
import cv2
import time

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
             detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[
                conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output


labelMap = [
    'armor',
    'base',
    'car',
    'target',
    'target-blue',
    'target-grey',
    'target-grey-2',
    'target-red',
    'watcher',
    'background'
]

cam_options = ['rgb', 'left', 'right']


def draw_boxes(frame, boxes, total_classes):
    if boxes is None or len(boxes) == 0:
        return frame
    else:

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = int(boxes[i, 0]), int(
                boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3])
            conf, cls = boxes[i, 4], int(boxes[i, 5])

            label = f"{labelMap[cls]}: {conf:.2f}"

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            # Get the width and height of label for bg square
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            # Shows the text.
            frame = cv2.rectangle(frame, (x1, y1 - 2*h),
                                  (x1 + w, y1), (0, 255, 0), -1)
            frame = cv2.putText(frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    return frame



model = torch.hub.load('ultralytics/yolov5', 'custom', 'apr29.pt', autoshape=True) 

# Use cv2 to run predictions for a video file, outputs out.avi, which has bounding box predictions
# you will have to paste video.avi for this part, which can be found on the google drive

cap = cv2.VideoCapture('video.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
writer = cv2.VideoWriter("video_annotated_old.avi", fourcc, fps, (320, 320), True)
i = 0

t = transforms.ToTensor()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break
     
    frame = cv2.resize(frame, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)

    frame_np = np.array(frame)

    x = t(frame_np).unsqueeze(0)

    pred = model(x)

    boxes = non_max_suppression(
            pred, 0.30, 0.45)
    
    boxes = boxes[0]

    if boxes is not None:
        frame = draw_boxes(frame, boxes.numpy(), 9)

    writer.write(frame)

    if i % 60 == 0:
        print(f'second {i // 60}')

    if i == 2000:
        break

    i += 1
cap.release()
writer.release()
