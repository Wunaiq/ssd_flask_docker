# coding=utf-8
from __future__ import division, print_function, absolute_import
import os
import sys

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from config import Config

app = Flask(__name__)
app.config.from_object(Config)


# SSD   
from SSDdetector import SSD_detctor

# define the model
model_path = "./app/SSDdetector/weights/ssd300_mAP_77.43_v2.pth"
model = SSD_detctor(model_path)

# print('Running on http://localhost:5000')

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.abspath(os.path.dirname(__file__))
    file_dir = os.path.join(basepath, './static/uploads',)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)  
    file_path = os.path.join(file_dir, secure_filename(f.filename)) 
     
    # file_path = os.path.abspath(os.path.join('./uploads', secure_filename(f.filename)))
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
        bboxs = model.detect(file_path)
        results_dir = "./static/detect-results/"
        resultImgSrc = results_dir + file_path.split('/')[-1]
        return resultImgSrc
    return None

if __name__ == '__main__':
    # Serve the app with gevent
    # http_server = WSGIServer(('', 5000), app)
    # http_server.serve_forever()
    app.run(debug=True, host='0.0.0.0', port=8008)