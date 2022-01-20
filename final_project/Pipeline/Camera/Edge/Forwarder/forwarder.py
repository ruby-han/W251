# import cv2 as cv
import numpy as np
import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST= "mosquitto-service" #'localhost'
LOCAL_MQTT_PORT= 1883
LOCAL_MQTT_TOPIC="faces_topic"

# NEW CODE TO ALLOW FORWARD TO REMOTE
REMOTE_MQTT_HOST = '35.182.93.9' # '99.79.78.134'# '3.19.64.123' # EC2 IP?
REMOTE_MQTT_PORT = 1883 #31813 # Jetson broker nodeport 
REMOTE_MQTT_TOPIC = "faces_topic"
# test lines

global counter 
counter = 0

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        client.subscribe(LOCAL_MQTT_TOPIC)

# NEW FUNCTION TO CONNECT TO REMOTE
def on_connect_remote(client, userdata, flags, rc):
        print("connected to remote broker with rc: " + str(rc))
        client.subscribe(REMOTE_MQTT_TOPIC)
	
def on_message(client,userdata, msg):


  try:
    #print("message received: ",str(msg.payload.decode("utf-8")))
    global counter
    counter +=1
    print("message received. bytes: ",len(msg.payload), "Count: ", counter)    
    # Above don't have to print out the message, can just count the number of messages received
    # if we wanted to re-publish this message, something like this should work
    msg = msg.payload
    remote_mqttclient.publish(REMOTE_MQTT_TOPIC, payload=msg, qos=0, retain=False)
    


    # Test out image decoding locally
    # data_encode = np.array(msg) # (msg.payload)
    # str_encode = data_encode.tostring()
    # nparr = np.fromstring(str_encode, np.uint8)
    # img_decode = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    # Debug
    # cv.imwrite(str(counter)+'.png', img_decode)
    # cv.imshow(str(counter), img_decode)
 
  except:
    print("Unexpected error (GL - in republishing):", sys.exc_info()[0])

remote_mqttclient = mqtt.Client()
remote_mqttclient.on_connect = on_connect_remote
remote_mqttclient.connect(REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, 60)

local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_message = on_message



# go into a loop
local_mqttclient.loop_forever()
# remote_mqttclient.loop_forever()
        

