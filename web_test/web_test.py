import matplotlib.pyplot as plt 
import numpy as np
import requests
import base64
import time
import sys
import cv2
import os
from io import BytesIO
import argparse
from PIL import Image
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
SUB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../app/SSDdetector"))
sys.path.append(BASE_DIR)
sys.path.append(SUB_DIR)


from app.SSDdetector.data import BaseTransform, VOCAnnotationTransform
from app.SSDdetector.data import VOC_CLASSES
from eval_on_web import eval_on_web


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class Test():
    def __init__(self, args):
        self.data_root = args.data_root
        self.restype = args.restype
        self.save_dir = args.save_dir
        self.test_url = args.test_url
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.cuda = args.cuda

    def test(self):
        
        print("Test...")

        _t = Timer()

        if self.restype == "bboxes":
            test_res_log = open(os.path.join(self.save_dir, "bboxes.json"), "wb")
        
        num = 0
        num_images = len(os.listdir(self.data_root))

        for img_name in os.listdir(self.data_root):
                
            img_path = os.path.join(self.data_root, img_name)
            with open(img_path, 'rb') as f:
                img = base64.b64encode(f.read())
            
            img_data = {'image': [img], 
                        'filename': img_name, 
                        'restype': self.restype,
                        'cuda': self.cuda}
            _t.tic()
            response = requests.post(self.test_url, data=img_data)
            detection_time = _t.toc(average=False)

            if self.restype == "image":
                image_res = plt.imread(BytesIO(response.content),"jpg")  # trans bytes to array
                image_res = image_res[:,:,::-1]   

                save_path = os.path.join(self.save_dir, img_name)
                cv2.imwrite(save_path, image_res)
            else:
                test_res_log.write(response.content)
                test_res_log.write(bytes("\n", encoding="utf-8"))

            num += 1
            print("test {}/{}, time: {:.4f}.".format(num, num_images, detection_time))

        if self.restype == "bboxes":
            test_res_log.close()

        print("Done!")
        total_time = _t.total_time
        fps = _t.calls / _t.total_time
        print('total_time: {:.3f}s, fps: {:.3f}s'.format(total_time, fps))


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # default set for test custom data
    # parser.add_argument("-restype", type=str, default="bboxes", help="return date type: image, bboxes or precision")
    # parser.add_argument("-data_root", type=str, default="./custom_data", help="test data")
    # parser.add_argument("-save_dir", type=str, default="./custom_results", help="test results save dir")
    # parser.add_argument("-test_url", type=str, default="http://0.0.0.0:8008/test", help="test url")
    # parser.add_argument("-cuda", type=str2bool, default=True, help="turn on gpu or turn off")
    
    # default set for test voc data
    parser.add_argument("-restype", type=str, default="precision", help="return date type: image, bboxes or precision")
    parser.add_argument("-data_root", type=str, default="../app/SSDdetector/data/VOCdevkit", help="test data")
    parser.add_argument("-save_dir", type=str, default="./voc_results", help="test results save dir")
    parser.add_argument("-test_url", type=str, default="http://0.0.0.0:8008/test", help="test url")
    parser.add_argument("-cuda", type=str2bool, default=True, help="turn on gpu or turn off")
    
    args = parser.parse_args()

    if args.restype == "precision":   # here use voc 2007 for precision test

        web_eval = eval_on_web(args.save_dir, args.data_root, YEAR='2007', cuda=args.cuda)
        web_eval.test_net(args.test_url)
    else:
        test = Test(args)
        test.test()
        
    print("Test results have been saved to {}".format(os.path.abspath(args.save_dir)))
    