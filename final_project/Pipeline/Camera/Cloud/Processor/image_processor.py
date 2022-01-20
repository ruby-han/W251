import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt
import sys
#import boto3
#from credentials import aws_access_key_id,aws_secret_access_key
#from botocore.exceptions import ClientError
#from botocore.config import Config

LOCAL_MQTT_HOST="0.0.0.0" # remove
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="faces_topic"
global counter
counter = 0

#my_config = Config(region_name = 'us-east-2')
#S3_client = boto3.client(
#    's3',
#    aws_access_key_id=aws_access_key_id,
#    aws_secret_access_key=aws_secret_access_key,
#    # aws_session_token=SESSION_TOKEN
#)

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)

def on_message(client,userdata, msg):
    global counter
#    global S3_client
    try:
        print("message received")#+str(type(msg))) # ,str(msg.payload.decode("utf-8")))
        # publish the message
        # remote_mqttclient.publish(REMOTE_MQTT_TOPIC,msg.payload)
        # if we wanted to re-publish this message, something like this should work

        binary_msg = msg.payload
        decoded_msg = np.frombuffer(binary_msg,dtype='uint8')
        img = cv.imdecode(decoded_msg, flags=1)

        cv.imwrite('face'+str(counter) + '.png', img)
#        try:
#            response = S3_client.upload_file('face'+str(counter) + '.png', 'gerritlensink-w251-hw3', 'face'+str(counter) + '.png')
#        except ClientError as e:
#            print(e)
        counter += 1
    except:
        print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()
