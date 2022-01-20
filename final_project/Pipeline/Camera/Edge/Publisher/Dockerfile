FROM ubuntu:latest

# this is needed to make sure we can see the log output
ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

RUN export OPENBLAS_CORETYPE=ARMV8
RUN apt-get update && apt-get install -y python3 python3-pip python3-opencv vim-tiny libopencv-dev

# install OpenCV, mqtt client, numpy
# RUN pip3 install paho-mqtt
RUN pip3 install numpy
RUN apt install -y mosquitto-clients
RUN pip3 install paho-mqtt
RUN pip3 install opencv-python

#RUN export DISPLAY=:0

# Copy face publisher file into the container
COPY face_publisher.py ./

# Copy xml file into container
COPY haarcascade_frontalface_default.xml ./

RUN chmod 700 *.*

CMD python3 /app/face_publisher.py
