#Flask-WebApp-With-MaskRCNN
Image Classification using Mask RCNN and Flask.
In this code, we used pretrained coco model and create Python Flask Web App.

we used pretrained Mask coco Tensorflow detection model. 
download pretrained model from : https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5 and save it into flask web app folder.

more: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Below is our requirements.txt file
mrcnn
Werkzeug
Flask
numpy
Keras
gevent
pillow
h5py
tensorflow

use the below command to load dependencies

$ pip install -r requirements.txt

The server can be started by running:

1. python app.py
2. Running on http://localhost:5000

Reference:
1. https://github.com/matterport/Mask_RCNN
2. https://github.com/mtobeiyf/keras-flask-deploy-webapp
3. https://www.pyimagesearch.com/2019/06/10/keras-mask-r-cnn/
