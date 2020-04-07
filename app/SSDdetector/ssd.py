from __future__ import division, print_function, absolute_import
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from matplotlib import pyplot as plt
from PIL import Image
import json
import time

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)

# from data import VOC_CLASSES as labels
from data import VOC_CLASSES as labels
from layers.ssd_model import build_ssd




class SSD_detctor():

    def __init__(self, model_path):
        
        self.net = build_ssd('test', 300, 21)    # initialize SSD
        self.net.load_weights(model_path)
        self.net.eval()

    def detect(self, input_image, cuda=True):
        """ input_image read as bgr image """

       
        image = cv2.resize(input_image, (300, 300)).astype(np.float32)
        image -= (104.0, 117.0, 123.0)
        image = image.astype(np.float32)
        image = image[:, :, ::-1].copy()     # BGR2RGB

        image = torch.from_numpy(image).permute(2, 0, 1)
        image = Variable(image.unsqueeze(0))     # wrap tensor in Variable

        if torch.cuda.is_available() and cuda == "True":
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.net = self.net.cuda()
            image = image.cuda()

        if cuda == "False":
            self.net = self.net.cpu()
            torch.set_default_tensor_type('torch.FloatTensor')

        outputs = self.net(image).data    # outputs [1, 21, 200, 5]
        
        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        # scale each detection back up to the input_image
        scale = torch.Tensor(input_image.shape[1::-1]).repeat(2)
        bboxs = []

        for i in range(outputs.size(1)):
            j = 0
            while outputs[0,i,j,0] >= 0.6:
                score = outputs[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (outputs[0,i,j,1:]*scale).cpu().numpy()
                bboxs.append(pt.tolist())
                # draw bbox
                cv2.rectangle(input_image, (pt[0], pt[1]), (pt[2], pt[3]), \
                                COLORS[i % 3], max(input_image.shape[0]//8000, 1))   
                cv2.putText(input_image, display_txt, (pt[0], pt[1]), FONT, \
                            max(input_image.shape[0]//1000, 0.4), COLORS[i % 3], \
                            max(input_image.shape[0]//800, 1), cv2.LINE_AA)
                j+=1

        return input_image, bboxs, outputs.cpu().numpy()

if __name__ == "__main__":

    model_path = "./weights/ssd300_mAP_77.43_v2.pth"
    model = SSD_detctor(model_path)

    img_dir = "./test_inputs"
    # results_dir = "./test_results"
    results_dir = "./test_results/images"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_log = open("./test_results/results_log.json", "w")


    start = time.time()
    num = 0
    for imgs in os.listdir(img_dir):
        imgs_path = os.path.join(img_dir, imgs)
        image = cv2.imread(imgs_path)
        
        res_image, res_bboxs = model.detect(image)

        cv2.imwrite(os.path.join(results_dir, imgs), res_image)
        results_log.write("image " + str(num) + " : " + imgs + "\n")
        json.dump(res_bboxs, results_log)
        results_log.write("\n")
        num += 1
    
    cost_time_total = time.time() - start
    print("Cost time totally: %.4f" % cost_time_total)
    print("Cost time per image: %.4f" % (cost_time_total / num))
    print("FPS is: %0.4f" % (num / cost_time_total))

    results_log.close()
