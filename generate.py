import os
import cv2
import numpy as np
import random

BASE_IMAGE_DIR = 'base-images'
LABELS_DIR = 'base-labels'
TRAIN_DIR = os.path.join('data', 'train')
VAL_DIR = os.path.join('data', 'val')
BG_DIR = 'backgrounds'

im_in = 0
la_in = 0
it = 0

train = 0.95

scale_min = 0.25
scale_max = 1.00

random.seed(1345)

#is_train = random.choices([0, 1], weights=[(1 - train), train], k=12802)

def apply_transforms(img, mask, x0, x1, y0, y1):

    # random scale 

    scale_factor = random.uniform(scale_min, scale_max)
    x0 = x0 * scale_factor
    x1 = x1 * scale_factor
    y0 = y0 * scale_factor
    y1 = y1 * scale_factor

    x0 = int(x0)
    x1 = int(x1)
    y0 = int(y0)
    y1 = int(y1)

    dsz = int(320 * scale_factor)

    mask = cv2.resize(mask, (dsz, dsz), scale_factor, scale_factor, interpolation=cv2.INTER_CUBIC)
    img = cv2.resize(img, (dsz, dsz), scale_factor, scale_factor, interpolation=cv2.INTER_CUBIC)

    shape = mask.shape[0]

    border_w = 320 - shape
    border_h = 320 - shape

    rand_w = random.randint(0, border_w)
    rand_h = random.randint(0, border_h)

    border_l = rand_w
    border_r = 320 - shape - rand_w
    border_t = rand_h
    border_b = 320 - shape - rand_h

    x0 += border_l
    x1 += border_l
    y0 += border_t
    y1 += border_t

    # always r, b
    mask = cv2.copyMakeBorder(mask, border_t, border_b, border_l, border_r, cv2.BORDER_CONSTANT, 255 )
    img = cv2.copyMakeBorder(img, border_t, border_b, border_l, border_r , cv2.BORDER_CONSTANT, 255)

    return img, mask, x0, x1, y0, y1

for image_loc in os.listdir(BASE_IMAGE_DIR):

    # open the corresponding label
    bbox_path = os.path.join(LABELS_DIR, image_loc.replace('.jpg', '.txt'))
    bbox = None
    with open(bbox_path, 'r') as f:
        bbox = f.readline().split(' ')
        _, x, y, w, h, = bbox
        x = float(x)
        y = float(y)
        w = float(w)
        h = float(h)
        la_in += 1
    

    x0 = int((x - (w / 2)) * 320)
    x1 = int((x + (w / 2)) * 320)
    y0 = int((y - (h / 2)) * 320)
    y1 = int((y + (h / 2)) * 320)
    
    img = cv2.imread(os.path.join(BASE_IMAGE_DIR, image_loc))
    
    # for each image synth_image_loc in synth dir, make a synth image and add the labels to label set
    i = 0

    for j in range(10):

        for bg_image_loc in os.listdir(BG_DIR):

            bg_image = cv2.imread(os.path.join(BG_DIR, bg_image_loc))

            small_green = np.array([0, 100, 0])     ##[R value, G value, B value] -> [B, G, R]
            big_green = np.array([100, 255, 235])

            mask = cv2.inRange(img, small_green, big_green)

            # apply transforms here
            new_img, mask, n_x0, n_x1, n_y0, n_y1 = apply_transforms(img, mask, x0, x1, y0, y1)

            res = cv2.bitwise_and(new_img, new_img, mask=mask)

            f = new_img - res
            f = np.where(f == 0, bg_image, f).astype(np.uint8)

            #cv2.rectangle(f, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #cv2.imwrite('output.jpg', f)
    
            if im_in == 1001:
                cv2.rectangle(f, (n_x0, n_y0), (n_x1, n_y1), (0, 255, 0), 2)
                cv2.imwrite('output.jpg', f)



            new_x = ((n_x0 + n_x1) / 2) / 320
            new_y = ((n_y0 + n_y1) / 2) / 320
            new_w = (abs(n_x1 - n_x0)) / 320 
            new_h = (abs(n_y1 - n_y0)) / 320

            save_loc = str(i) + "_" + image_loc
            if 'jpeg' in image_loc:
                label_name = save_loc.replace('.jpeg', '.txt')
            else:
                label_name = save_loc.replace('.jpg', '.txt')  

            cv2.imwrite(os.path.join('data', 'train', 'images', save_loc), f)
        
            with open(os.path.join('data', 'train', 'labels', label_name), 'w') as f:
                f.write(f'0 {new_y} {new_x} {new_h} {new_w}\n')

            i += 1
            it += 1

            if im_in % 50 == 0:
                print(f'{im_in} images processed')
            im_in += 1
        

print(f'{im_in} images found, {la_in} labels loaded') 
#print(f'{synth} synthetic images made')
