import os
import easyocr
import cv2
import random
import numpy as np

BASE_IMAGE_DIR = 'base-images'
LABELS_DIR = 'base-labels'
TRAIN_DIR = os.path.join('data', 'train')
VAL_DIR = os.path.join('data', 'val')
BG_DIR = 'backgrounds'

random_bg_image = os.path.join(BG_DIR, '11.jpg')
random_bg_image = cv2.imread(random_bg_image)

reader = easyocr.Reader(['en'])

im_in = 0
la_in = 0
it = 0

train = 0.95
is_train = random.choices([0, 1], weights=[(1 - train), train], k=12802)

for img_loc in os.listdir(BASE_IMAGE_DIR):
  


