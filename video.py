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

import os
import pathlib
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom', 'apr22.pt', autoshape=False) 

# Use cv2 to run predictions for a video file, outputs out.avi, which has bounding box predictions
# you will have to paste video.avi for this part, which can be found on the google drive

cap = cv2.VideoCapture('video.avi')
writer = None
i = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break
     
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame = cv2.resize(frame, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)

    frame_np = np.array(frame)

    pred = model(frame_np.reshape(1, 320, 320, 3) / 255.0)

    print(pred)
    x0, y0, x1, y1 = pred[0][0], pred[0][1], pred[0][2], pred[0][3]

    x0 -= 50
    x1 -= 50
    y0 += 20
    y1 += 20
    
    x0 = int(x0)
    x1 = int(x1)
    y0 = int(y0)
    y1 = int(y1)

    """

    print(x0)
    print(x1)
    print(y0)
    print(y1)
    print()

    """

    cv2.rectangle(frame_np,(x0,y0),(x1,y1),(0,255,0),2)
    i += 1
  
    if writer == None:

        writer = cv2.VideoWriter("out.avi", fourcc, fps, (320, 320), True)
        print(writer)
    else:
        print("writer is not none")
        print(writer)

    writer.write(frame_np)
cap.release()
writer.release()
