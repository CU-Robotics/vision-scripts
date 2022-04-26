import cv2
import os

IMAGE_DIR = 'synth_bg'
SIZE = 320

for image_loc in os.listdir(IMAGE_DIR):

    img = cv2.imread(os.path.join(IMAGE_DIR, image_loc))

    img = cv2.resize(img, (SIZE, SIZE))

    cv2.imwrite(os.path.join(IMAGE_DIR, image_loc), img)


