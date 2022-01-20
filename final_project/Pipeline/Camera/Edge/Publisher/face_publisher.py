# Import packages
import cv2 as cv
import numpy as np
import paho.mqtt.client as mqtt

########## SETTING UP THE MQTT ##########
LOCAL_MQTT_HOST = 'mosquitto-service' # for docker only: '0.0.0.0'  
LOCAL_MQTT_PORT = 1883 # for Docker Only: 32003
LOCAL_MQTT_TOPIC = 'faces_topic'

# Function to veryfy connection
def on_connect_local(client, userdata, flags, rc):
    print('connected to local broker with rc: ' + str(rc))
    
# Set up the connection
local_mqttclient = mqtt.Client()
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)    # Turn this off for the code to run w/o connection


########## FACE DETECTION ##########

# Load XML classifiers for detecting face and eyes
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#face_cascade = cv.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_eye.xml')
# Works in version 4.5.3:
#face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')


# Start the video capture from port 0
cap = cv.VideoCapture(0)

# Track frame number for saving images
frame_num = 0

# Keep running until 'q' pressed
while(True):
    
    # Capture frame by frame from the webcam
    ret, frame = cap.read()
    
    # Identify all of the faces in the frame
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)    # params: image, scaleFactor (step size), minNeighbors (quality v detection ability)
    
    # For each face identified, draw a box, save the face
    for count, (x,y,w,h) in enumerate(faces): 
        cv.rectangle(frame, (x,y),(x+w, y+h), (255, 0, 0), 3)
        
        # Debug - Grab just the face, save it    
        face = frame[y:y+h, x:x+w]
        facenum = ("Face_"+ str(count))
	
        # debug     
        #cv.imwrite("face_tests/"+str(frame_num)+"_"+facenum+".jpg", face)
        #cv.imwrite("../face_tests/"+str(frame_num)+"_"+facenum+".jpg", face)

	
        # Save the face as binary 
        face_bytes = cv.imencode('.jpg', face)[1].tobytes()
        
        # Publish to queue
        #local_mqttclient.publish(LOCAL_MQTT_TOPIC, str(facenum)+str(frame_num))
        local_mqttclient.publish(LOCAL_MQTT_TOPIC, face_bytes)
        
    # Display the frame captured
    #cv.imshow('frame', frame)    

    frame_num += 1
    if cv.waitKey(1000) & 0xFF == ord('q'): # waitKey = 1000 = new frame every 1000ms
        break
        
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
