# SSD Flask Docker

A SSD object detection web project implemented with Flask and Docker.

The referenced paper is: SSD: Single Shot MultiBox Detector https://arxiv.org/pdf/1512.02325.pdf


## Main Requirements and Versions

```
python 3.5.5
pytorch 1.2.0
cuda 9.0
cudnn 7
```


## Quick Start

### 1. Prepare the environment

Install dependency packages according to the requirements.txt:
```
pip3 install -r requirements.txt
```

For convenience, you can also use the docker images has been build as following:

#### Pull the pre-built docker image

```
docker pull wunaiq/ssd_flask_docker:cuda90_cudnn7_py35_pytorch1.12_ubuntu16.04
```


#### Run the docker container

```
docker run -it --gpus all -v /base_path/ssd_flask_docker:/ssd_flask_docker -p 8008:8008 ssd_flask_docker:cuda90_cudnn7_py35_pytorch1.12_ubuntu16.04 /bin/bash
```
Please replace the /base_path in the above command with the absolute path of your root directory.

Args in the above command:
- -v: link the folder of host (/base_path/ssd_flask_docker) with a folder of container(/ssd_flask_docker)

- -p: map the port 8008 of container to the port 8008 of host. We has set the sever port as 8008 in ./app/app.py 

### 2. Run the web server

The server can be started by running:

```
cd ./ssd_flask_docker
python3 app/app.py
```

Now the server is running on: http://0.0.0.0:8008

Tips:
- Download model weights at: https://drive.google.com/open?id=1eld9vbMju-79cX3Lw6oxhoHR3aAmIkSj
- The model file should be put in ``` ./app/SSDdetector/weights/ ``` 
- The uploads files and detector results are in ``` ./app/static/ ```

### 3. Run a demo of SSD or debug the model alone
```
cd ./app/SSDdetector
python3 ssd_model.py
```

## SSD Training

### 1. Preparation

Download COCO:

```
cd ./SSDdetector
./data/scripts/COCO2014.sh
```

Download VOC:

```
./data/scripts/VOC2007.sh
./data/scripts/VOC2012.sh
```

Download the fc-reduced VGG-16 PyTorch base network weights:
```
mkdir weights
cd ./weights
wget https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth

```


### 2. Trian 

```
cd ./app/SSDdetector/
python3 ./trainval/train.py
```

*Training loss visualization with visdom:
```
pip3 install visdom
python3 -m visdom.server
# Then navigate to http://localhost:8097/ during training
```



## Eval

```
cd ./app/SSDdetector/
python3 ./trainval/eval.py
```

Performance on VOC2007 Test：

##### mAP

| Original | Converted weiliu89 weights | From scratch w/o data aug | From scratch w/ data aug |
|:-:|:-:|:-:|:-:|
| 77.2 % | 77.26 % | 58.12% | 77.43 % |

##### FPS
**GTX 1060:** ~45.45 FPS



&nbsp;

&nbsp;

&nbsp;

&nbsp;


---

### Reference:

1. Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector[C]//European conference on computer vision. Springer, Cham, 2016: 21-37.
2. https://github.com/amdegroot/ssd.pytorch.git
3. https://github.com/jomalsan/pytorch-mask-rcnn-flask.git
4. https://github.com/imadelh/ML-web-app.git
