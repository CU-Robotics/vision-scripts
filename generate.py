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

random.seed(12345)

#is_train = random.choices([0, 1], weights=[(1 - train), train], k=12802)

def apply_transforms(mask, x0, x1, y0, y1):

    # random scale 

    scale_factor = random.uniform(scale_min, scale_max)
    x0 = x0 * scale_factor
    x1 = x1 * scale_factor
    y0 = y0 * scale_factor
    y1 = x1 * scale_factor

    dsz = int(320 * scale_factor)

    mask = cv2.resize(mask, (dsz, dsz), scale_factor, scale_factor, interpolation=cv2.INTER_CUBIC)

    shape = mask.shape[0]

    # always r, b
    mask = cv2.copyMakeBorder(mask, 0, (320 - shape), 0, (320 - shape), cv2.BORDER_CONSTANT, value=255)

    return mask, x0, x1, y0, y1

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
    

    #x, y, wc, hc
    #cls = 0
    #x = ((x0 + x1) / 2) / 320
    #y = ((y0 + y1) / 2) / 320
    #w = (abs(x1 - x0)) / 320 
    #h = (abs(y1 - y0)) / 320


    x0 = int((x - (w / 2)) * 320)
    x1 = int((x + (w / 2)) * 320)
    y0 = int((y - (h / 2)) * 320)
    y1 = int((y + (h / 2)) * 320)
    
    img = cv2.imread(os.path.join(BASE_IMAGE_DIR, image_loc))
    
    #cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 2)

    cv2.imwrite('test.jpg',img)

    # for each image synth_image_loc in synth dir, make a synth image and add the labels to label set
    i = 0

    for bg_image_loc in os.listdir(BG_DIR):

        bg_image = cv2.imread(os.path.join(BG_DIR, bg_image_loc))

        small_green = np.array([0, 100, 0])     ##[R value, G value, B value] -> [B, G, R]
        big_green = np.array([100, 255, 235])

        mask = cv2.inRange(img, small_green, big_green)


        print(mask.shape)
        # apply transforms here
        mask, x0, x1, y0, y1 = apply_transforms(mask, x0, x1, y0, y1)

        print(mask.shape)

        res = cv2.bitwise_and(img, img, mask=mask)

        f = img - res
        f = np.where(f == 0, bg_image, f).astype(np.uint8)

        cv2.imwrite('output.jpg', f)
        break
        """
            try:
            bg = cv2.imread(os.path.join(BG_DIR, bg_image_loc))

            bg[x0:x1, y0:y1] = img[x0:x1, y0:y1]

            if is_train[i] == 1:
                save_loc = str(i) + "_" + image_loc
                if 'jpeg' in image_loc:
                    label_name = save_loc.replace('.jpeg', '.txt')
                else:
                    label_name = save_loc.replace('.jpg', '.txt')  

                cv2.imwrite(os.path.join('train', 'images', save_loc), bg)
            
                with open(os.path.join('train', 'labels', label_name), 'w+') as f:
                    f.write(f'{cls} {y} {x} {h} {w}\n')

            else:
                save_loc = str(i) + "_" + image_loc
                if 'jpeg' in image_loc:
                    label_name = save_loc.replace('.jpeg', '.txt')
                else:
                    label_name = save_loc.replace('.jpg', '.txt')  

                cv2.imwrite(os.path.join('valid', 'images', save_loc), bg)
            
                with open(os.path.join('valid', 'labels', label_name), 'w+') as f:
                    f.write(f'{cls} {y} {x} {h} {w}\n')
            i += 1
            it += 1
        except Exception as e:
            print(e)
            pass
        """

        if im_in % 50 == 0:
            print(f'{im_in} images processed')

        im_in += 1
    

print(f'{im_in} images found, {la_in} labels loaded') 
#print(f'{synth} synthetic images made')
