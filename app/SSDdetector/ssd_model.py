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


# from data import VOC_CLASSES as labels
from .data import VOC_CLASSES as labels

from .ssd import build_ssd

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)


class SSD_detctor():

    def __init__(self, model_path):
        
        self.net = build_ssd('test', 300, 21)    # initialize SSD
        self.net.load_weights(model_path)
        

    def detect(self, input_image_path):
        """ input_image read by cv2 """
        
        rbg_image = cv2.imread(input_image_path)
        rgb_image = cv2.cvtColor(rbg_image, cv2.COLOR_BGR2RGB)
        
        x = cv2.resize(rgb_image, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()

        x = torch.from_numpy(x).permute(2, 0, 1)
        # x = torch.from_numpy(x)  
        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = self.net(xx)

        top_k=10

        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        bboxs = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.2:
                score = detections[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                bboxs.append(pt.tolist())
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # color = colors[i]
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                cv2.rectangle(rbg_image, (pt[0], pt[1]), (pt[2], pt[3]), COLORS[i % 3], 1)
                cv2.putText(rbg_image, display_txt, (pt[0], pt[1]), FONT, 0.4, COLORS[i % 3], 1)
                j+=1

        # res_img = rbg_image.tolist()

        # res_dict = {}
        # res_dict['img'] = res_img
        # res_dict['bboxs'] = bboxs
        # res_dict['result'] = 'sample'

        # res_json = json.dumps(res_dict)
        # with open("json.txt", "w") as f:
        #     f.write(res_json)
        img_name = input_image_path.split('/')[-1]
        print(img_name)

        results_dir = "./static/detect-results/"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        cv2.imwrite(results_dir + img_name, rbg_image)
        return bboxs

if __name__ == "__main__":

    model_path = "./SSD_app/SSDdetector/weights/ssd300_mAP_77.43_v2.pth"

    detector = SSD_detctor(model_path)
    # img = cv2.imread("01.jpg")
    img = './01.jpg'
    res = detector.detect(img)
    # print(res)