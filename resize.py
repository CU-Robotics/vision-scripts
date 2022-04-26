import cv2
import os

BASE_IMAGE_DIR = 'base-images'
BG_IMAGE_DIR = 'backgrounds'
SIZE = 320

for image_loc in os.listdir(BASE_IMAGE_DIR):
    img = cv2.imread(os.path.join(BASE_IMAGE_DIR, image_loc))
    img = cv2.resize(img, (SIZE, SIZE))
    cv2.imwrite(os.path.join(BASE_IMAGE_DIR, image_loc), img)

for image_loc in os.listdir(BG_IMAGE_DIR):
    img = cv2.imread(os.path.join(BG_IMAGE_DIR, image_loc))
    img = cv2.resize(img, (SIZE, SIZE))
    cv2.imwrite(os.path.join(BG_IMAGE_DIR, image_loc), img)


