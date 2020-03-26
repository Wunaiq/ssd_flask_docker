FROM dwqy11/ssd_flask:2

# for flask web server
EXPOSE 5000

# set working directory
# ADD . /SSD_app
WORKDIR /SSD_Flask_docker

# install required libraries
COPY ./ ./

# This is the runtime command for the container
CMD bin/bash 
