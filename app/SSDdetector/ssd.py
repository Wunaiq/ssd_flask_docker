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
from tqdm import tqdm

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)


# from data import VOC_CLASSES as labels
from data import VOC_CLASSES as labels

from layers.ssd_model import build_ssd

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

# module_path = os.path.abspath(os.path.join('..'))
# if module_path not in sys.path:
#     sys.path.append(module_path)


class SSD_detctor():

    def __init__(self, model_path):
        
        self.net = build_ssd('test', 300, 21)    # initialize SSD
        self.net.load_weights(model_path)
        

    def detect(self, image):
        """ input_image read by cv2 """
        # if not image:
        #     raise "The image is not exist!"
        # rbg_image = cv2.imread(input_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        rgb_image = cv2.resize(rgb_image, (300, 300)).astype(np.float32)
        rgb_image -= (104.0, 117.0, 123.0)
        rgb_image = rgb_image.astype(np.float32)
        rgb_image = rgb_image[:, :, ::-1].copy()

        rgb_image = torch.from_numpy(rgb_image).permute(2, 0, 1)
        # x = torch.from_numpy(x)  
        rgb_image = Variable(rgb_image.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            rgb_image = rgb_image.cuda()
        outputs = self.net(rgb_image).data

        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        FONT = cv2.FONT_HERSHEY_SIMPLEX

        # scale each detection back up to the image
        scale = torch.Tensor(image.shape[1::-1]).repeat(2)
        bboxs = []
        for i in range(outputs.size(1)):
            j = 0
            while outputs[0,i,j,0] >= 0.2:
                score = outputs[0,i,j,0]
                label_name = labels[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (outputs[0,i,j,1:]*scale).cpu().numpy()
                bboxs.append(pt.tolist())
                # coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                # color = colors[i]
                # currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                # currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                cv2.rectangle(image, (pt[0], pt[1]), (pt[2], pt[3]), COLORS[i % 3], 1)
                cv2.putText(image, display_txt, (pt[0], pt[1]), FONT, 0.4, COLORS[i % 3], 1)
                j+=1

        # res_img = rbg_image.tolist()

        # res_dict = {}
        # res_dict['img'] = res_img
        # res_dict['bboxs'] = bboxs
        # res_dict['result'] = 'sample'

        # res_json = json.dumps(res_dict)
        # with open("json.txt", "w") as f:
        #     f.write(res_json)
        # img_name = input_image_path.split('/')[-1]
        # print(img_name)

        # results_dir = "./app/static/detect-results/"
        # print(results_dir)
        # if not os.path.exists(results_dir):
        #     os.makedirs(results_dir)
        # cv2.imwrite(results_dir + img_name, rbg_image)
        
        return image, bboxs

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
    for imgs in tqdm(os.listdir(img_dir)):
        imgs_path = os.path.join(img_dir, imgs)
        image = cv2.imread(imgs_path)
        
        res_image, res_bboxs = model.detect(image)

        cv2.imwrite(os.path.join(results_dir, imgs), res_image)
        results_log.write("image " + str(num) + " : " + imgs + "\n")
        json.dump(res_bboxs, results_log, "a")
        results_log.write("\n")
        num += 1
    
    cost_time_total = time.time() - start
    print("Cost time totally: %.4f" % cost_time_total)
    print("Cost time per image: %.4f" % (cost_time_total / num))
    print("FPS is: %0.4f" % (num / cost_time_total))

    results_log.close()
