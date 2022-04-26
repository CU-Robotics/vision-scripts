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
    img = cv2.imread(os.path.join(BASE_IMAGE_DIR, img_loc))

    small_green = np.array([0, 50, 0])     ##[R value, G value, B value] -> [B, G, R]
    big_green = np.array([100, 255, 235])

    mask = cv2.inRange(img, small_green, big_green)
    res = cv2.bitwise_and(img, img, mask = mask)


    f = img - res
    f = np.where(f == 0, random_bg_image, f).astype(np.uint8)

    #cv2.imshow('img', f)
    #cv2.waitKey(0)
    cv2.imwrite('output.jpg', f)


