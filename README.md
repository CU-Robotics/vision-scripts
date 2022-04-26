### BuffBot Vision Scripts

This repo was meant to hold all of the scripts that we use to train, test and iterate on our computer vision system. As of the 2022 season, we are using the [YOLOv5 model](https://github.com/ultralytics/yolov5) to perform object detection on the [Luxonis OAK-D](https://docs.luxonis.com/en/latest/). While the models are trained in pytorch, since the forward pass is performed on the camera, we have to translate the weights to a weird OpenVINO blob format. [Here is a colab notebook that does that.](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb)

### Stuff you can do in this repo

##### Run detections directly on the depthai camera
1. Get your model blob from google drive (Software/Models/Best.blob), put it into ```weights/```
2. Run ```luxonis_demo.py```. it should spawn an openCV window showing a live video stream from the camera, with overlaid bounding boxes.

##### Generate Synthetic data


##### Load and save models from AWS S3 (WIP)
1. Make sure you have the aws credentials ```cred

