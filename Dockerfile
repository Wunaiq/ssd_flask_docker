FROM anibali/pytorch:cuda-9.0

# for flask web server
EXPOSE 8008

# set working directory
WORKDIR /ssd_flask_docker

# install required libraries
COPY . .
RUN sudo sed -i 's/archive.ubuntu.com/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && sudo apt-get update \
    && sudo apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && pip install setuptools -i https://mirrors.ustc.edu.cn/pypi/web/simple \
    && pip install -r requirements.txt -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
    
# This is the runtime command for the container
CMD python app/app.py
