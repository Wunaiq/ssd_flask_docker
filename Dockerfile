FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04 

RUN sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt-get clean \
    && apt-get update \
    && apt-get install -yq --no-install-recommends \
    python3.5 \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && pip3 install setuptools -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# for flask web server
EXPOSE 8008

# set working directory
WORKDIR /ssd_flask_docker

# install required libraries
COPY requirements.txt ./
RUN pip3 install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    
# This is the runtime command for the container
CMD bin/bash
