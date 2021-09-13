import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt
import sys
import boto3
from credentials import aws_access_key_id, aws_secret_access_key
from botocore.config import Config
from botocore.exceptions import ClientError

LOCAL_MQTT_HOST="mosquitto-service" # "0.0.0.0" 
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="face_detection"

count = 0

my_config = Config(region_name = 'ca-central-1')
S3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)
	
def on_message(client,userdata, msg):
    global count
    global S3_client
    try:
        print("message received")#+str(type(msg))) # ,str(msg.payload.decode("utf-8")))
        # publish the message
        # remote_mqttclient.publish(REMOTE_MQTT_TOPIC,msg.payload)
        # if we wanted to re-publish this message, something like this should work
        msg = msg.payload
        decode = np.frombuffer(msg,dtype='uint8')
        img = cv.imdecode(decode, flags=1)#cv.IMREAD_COLOR)
        
        cv.imwrite(f'face_{count}.png', img)
        try:
            response = S3_client.upload_file(f'face_{count}.png', 
            'rubyhan-w251-hw3', f'face_{count}.png')
        except ClientError as e:
            print(e)
        count+=1
    except:
        print("Unexpected error:", sys.exc_info()[0])

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message

# go into a loop
local_mqttclient.loop_forever()