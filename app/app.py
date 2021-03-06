# coding=utf-8
from __future__ import division, print_function, absolute_import
import os
import sys
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
import cv2
import time
import json
import base64
import numpy as np
from PIL import Image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, Response, make_response, jsonify
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from config import Config

app = Flask(__name__)
app.config.from_object(Config)


# SSD   
from SSDdetector.ssd import SSD_detctor

# define the model
model_path = "./app/SSDdetector/weights/ssd300_mAP_77.43_v2.pth"
model = SSD_detctor(model_path)

# print('Running on http://localhost:5000')

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.abspath(os.path.dirname(__file__))
    file_dir = os.path.join(basepath, Config.UPLOAD_IMAGES_DIR,)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)  
    file_path = os.path.join(file_dir, secure_filename(f.filename)) 
     
    f.save(file_path)
    return file_path


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        
        # image_name = file_path.split('\\')[-1]   # on windows
        image_name = file_path.split('/')[-1]   # on linux

        image = cv2.imread(file_path, cv2.IMREAD_COLOR)
        res_image, res_bboxs, _ = model.detect(image)
        
        basepath = os.path.abspath(os.path.dirname(__file__))
        res_dir = os.path.join(basepath, Config.DETECTION_RESULTS_DIR)
        res_log = os.path.join(basepath, Config.DETECTION_RESULTS_LOG)
        # save detection image result
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)
        cv2.imwrite(os.path.join(res_dir, image_name), res_image)
        
        # write detection bboxs log
        with open(res_log, "a") as f:
            time_stamp = int(round(time.time()*1000))
            time_stamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time_stamp/1000))

            f.write(time_stamp + "  " + image_name + "\n")
            json.dump(res_bboxs, f)
            f.write("\n")            

        return os.path.join(Config.DETECTION_RESULTS_DIR, image_name)  
    return None

@app.route('/test', methods=['POST', 'GET'])
def test():
    image_data = base64.b64decode(str(request.form['image']))
    image_name = str(request.form['filename'])
    restype = str(request.form['restype'])
    cuda = str(request.form['cuda'])
    image_data = np.fromstring(image_data, np.uint8)
    image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

    res_image, res_bboxs, outputs = model.detect(image, cuda)

    if restype == "precision":
        outputs = np.ndarray.tolist(outputs)  # numpy ndarry to list
        outputs_dict = {}
        outputs_dict['outputs'] = outputs
        return jsonify(outputs_dict)
        
    elif restype == "bboxes":
        res_bboxs_dict = {}
        res_bboxs_dict[image_name] = res_bboxs
        return jsonify(res_bboxs_dict)

    else:          
        _, buf = cv2.imencode(".jpg", res_image)
        res_image = Image.fromarray(np.uint8(buf)).tobytes()
        return res_image
    

if __name__ == '__main__':
    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()
    app.run(debug=True, host='0.0.0.0', port=8008)