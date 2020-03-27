import os
# basedir = os.path.abspath(os.path.dirname(__file__))

class Config(object):
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    DEBUG = True

    UPLOAD_IMAGES_DIR = "./static/uploads"
    DETECTION_RESULTS_DIR = "./static/detection_results/images"
    DETECTION_RESULTS_LOG = "./static/detection_results/detection_results_log.txt"