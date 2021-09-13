import numpy as np
import cv2 as cv
import paho.mqtt.client as mqtt

LOCAL_MQTT_HOST="mosquitto-service" #"0.0.0.0"
LOCAL_MQTT_PORT=1883 #30368 
LOCAL_MQTT_TOPIC="face_detection"

def on_connect_local(client, userdata, flags, rc): 
    print("connected to local broker with rc: " + str(rc))
 
local_mqttclient = mqtt.Client()
local_mqttclient.loop_start()
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)
local_mqttclient.on_connect = on_connect_local

cap = cv.VideoCapture(0)#, cv.CAP_V4L2)
face_cascade = cv.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('/usr/share/opencv4/haarcascades/haarcascade_eye.xml')

while True:
	ret, img = cap.read()
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_cascade.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
			cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		msg = cv.imencode('.png', roi_color)[1].tobytes()

		local_mqttclient.publish(LOCAL_MQTT_TOPIC, payload=msg,qos=0,retain=False)
        # print("Message Sent")

	cv.imshow('frame',img)
	if cv.waitKey(1) & 0xFF == ord('q'):
		break

# Release capture when done
cap.release()
cv.destroyAllWindows()