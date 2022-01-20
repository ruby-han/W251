FROM ubuntu:latest

ENV DEBIAN_FRONTEND noninteractive

# this is needed to make sure we can see the log output
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install pip
RUN apt-get update && apt-get install -y python3 python3-pip # python3-opencv

#install MQTT mosquitto client
RUN pip3 install paho-mqtt
# RUN pip3 install opencv-python
RUN pip3 install numpy

# Copy your listener.py file into the container
COPY forwarder.py ./

CMD python3 forwarder.py
