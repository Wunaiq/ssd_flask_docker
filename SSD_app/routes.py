# coding=utf-8
from __future__ import division, print_function
import os

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from SSD_app import app

# SSD
from .SSDdetector import SSD_detctor

# define the model
model_path = "./SSD_app/SSDdetector/weights/ssd300_mAP_77.43_v2.pth"
objdet_model = SSD_detctor(model_path)

# print('Running on http://localhost:5000')

def get_file_path_and_save(request):
    # Get the file from post request
    f = request.files['file']

    # Save the file to ./uploads
    basepath = os.path.dirname(__file__)
    file_path = os.path.join(
        basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)
    return file_path


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predictcoco', methods=['GET', 'POST'])
def predictcoco():
    if request.method == 'POST':
        file_path = get_file_path_and_save(request)
        bboxs = objdet_model.detect(file_path)
        resultImgSrc = "static/detect-results/" + file_path.split('\\')[-1]
        return resultImgSrc
    return None

if __name__ == '__main__':
    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
    # app.run(debug=True)