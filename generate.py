import os
import easyocr
import cv2
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
#is_train = random.choices([0, 1], weights=[(1 - train), train], k=12802)

for image_loc in os.listdir(IMAGE_DIR):

    print(f'{it} images processed')

    bbox = 

    result = reader.readtext(os.path.join(IMAGE_DIR, image_loc))

    # read the 
    im_in += 1
    if result != []:

        result = result[0][0]
        la_in += 1
        x0, y0 = result[0]
        x1, y1 = result[2]

        x0 -= 25
        y0 -= 15
        x1 += 25
        y1 += 15

        #x, y, wc, hc
        cls = 0
        x = ((x0 + x1) / 2) / 320
        y = ((y0 + y1) / 2) / 320
        w = (abs(x1 - x0)) / 320 
        h = (abs(y1 - y0)) / 320
        
        img = cv2.imread(os.path.join(IMAGE_DIR, image_loc))
        
        #cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 5)

        #cv2.imwrite('test.jpg',img)

        # for each image synth_image_loc in synth dir, make a synth image and add the labels to label set
        i = 0


        for bg_image_loc in os.listdir(BG_DIR):
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
        

print(f'{im_in} images found, {la_in} labels loaded') 
#print(f'{synth} synthetic images made')
