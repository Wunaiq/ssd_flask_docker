"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import sys
import os
import time
import argparse
from PIL import Image
import numpy as np
import pickle
import cv2
import base64
import requests
import json

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data

from app.SSDdetector.data import VOCAnnotationTransform, BaseTransform
from app.SSDdetector.data import VOC_CLASSES as labelmap

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

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


class VOC_dataload():
    ''' 
    load voc data for test without pytorch dataloader
    data_root is the path of VOCdevkit
    '''
    def __init__(self, root,
                 image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                 transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.transform = transform
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, gt, _, _ = self.pull_item(index)

        return im, gt

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()
        img = cv2.imread(self._imgpath % img_id)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)

        if self.transform is not None:
            target = np.array(target)
            img, boxes, labels = self.transform(img, target[:, :4], target[:, 4])
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))
        return img, target, height, width

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        # return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        return cv2.imread(self._imgpath % img_id)

    def pull_anno(self, index):
        '''Returns the original annotation of image at index

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to get annotation of
        Return:
            list:  [img_id, [(label, bbox coords),...]]
                eg: ('001718', [('dog', (96, 13, 438, 332))])
        '''
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot()
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

class eval_on_web():
    def __init__(self, save_dir, voc_root, YEAR):
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        self.annopath = os.path.join(voc_root, 'VOC' + YEAR, 'Annotations', '%s.xml')
        self.imgpath = os.path.join(voc_root, 'VOC' + YEAR, 'JPEGImages', '%s.jpg')
        self.imgsetpath = os.path.join(voc_root, 'VOC' + YEAR, 'ImageSets',
                                'Main', '{:s}.txt')                      # on linux
        
        self.dataset_mean = (104, 117, 123)
        self.set_type = 'test'

        # load data
        dataset_mean = (104, 117, 123)
        self.dataset = VOC_dataload(voc_root, [('2007', 'test')],
                            BaseTransform(300, dataset_mean),
                            VOCAnnotationTransform())

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                                int(bbox.find('ymin').text) - 1,
                                int(bbox.find('xmax').text) - 1,
                                int(bbox.find('ymax').text) - 1]
            objects.append(obj_struct)

        return objects


    def get_output_dir(self, save_dir, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = os.path.join(save_dir, name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir


    def get_voc_results_file_template(self, image_set, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + image_set + '_%s.txt' % (cls)
        filedir = os.path.join(self.save_dir, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path


    def write_voc_results_file(self, all_boxes, dataset):
        for cls_ind, cls in enumerate(labelmap):
            print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(self.set_type, cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dataset.ids):
                    dets = all_boxes[cls_ind+1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                    dets[k, 0] + 1, dets[k, 1] + 1,
                                    dets[k, 2] + 1, dets[k, 3] + 1))


    def do_python_eval(self, output_dir='output', use_07=True):
        cachedir = os.path.join(self.save_dir, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(labelmap):
            filename = self.get_voc_results_file_template(self.set_type, cls)
            rec, prec, ap = self.voc_eval(
            filename, self.annopath, self.imgsetpath.format(self.set_type), cls, cachedir,
            ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')


    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


    def voc_eval(self, 
                detpath,
                annopath,
                imagesetfile,
                classname,
                cachedir,
                ovthresh=0.5,
                use_07_metric=True):
        """rec, prec, ap = voc_eval(detpath,
                            annopath,
                            imagesetfile,
                            classname,
                            [ovthresh],
                            [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
    detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
    annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
    (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(annopath % (imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)

        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                    'difficult': difficult,
                                    'det': det}

        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:

            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]

            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                        (BBGT[:, 2] - BBGT[:, 0]) *
                        (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)

                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.

            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.

        return rec, prec, ap


    # def test_net(save_folder, net, cuda, dataset, transform, top_k,
    #              im_size=300, thresh=0.05):
    def test_net(self, test_url):
        num_images = len(self.dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)]
                    for _ in range(len(labelmap)+1)]    # [len(labelmap)+1, num_images, 0]

        # timers
        _t = {'im_detect': Timer(), 'misc': Timer()}
        output_dir = self.get_output_dir(self.save_dir, 'pkls', self.set_type)
        det_file = os.path.join(output_dir, 'detections.pkl')

        for i in range(num_images):
            im, gt, h, w = self.dataset.pull_item(i)

            _, buf = cv2.imencode(".jpg", im)
            img_bin = Image.fromarray(np.uint8(buf)).tobytes()
            img_bin = base64.b64encode(img_bin)
            img_data = {'image': [img_bin], 
                        'filename': i+1,
                        'restype': "precision"}

            _t['im_detect'].tic()
            response = requests.post(test_url, data=img_data)
            detect_time = _t['im_detect'].toc(average=False)

            detections = str(response.content, encoding='utf-8')
            detections = eval(detections)['outputs']
            detections = np.array(detections)
            detections = torch.from_numpy(detections)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):  # detections:[batch_size, num_classes, top_k, 5]
                dets = detections[0, j, :]
                # print(dets.shape)
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)  
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                    scores[:, np.newaxis])).astype(np.float32,
                                                                    copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time))
        
        # total_time = time.time() - start
        # print('total_time: {:.3f}s, fps: {:.3f}s'.format(total_time, num_images/total_time))
        total_time = _t['im_detect'].total_time
        fps = _t['im_detect'].calls / _t['im_detect'].total_time
        print('total_time: {:.3f}s, fps: {:.3f}s'.format(total_time, fps))

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        self.evaluate_detections(all_boxes, output_dir, self.dataset)


    def evaluate_detections(self, box_list, output_dir, dataset):
        self.write_voc_results_file(box_list, self.dataset)
        self.do_python_eval(output_dir)


