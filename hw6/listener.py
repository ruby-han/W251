import numpy as np
import cv2

camSet = 'udpsrc port=5000 ! application/x-rtp, media=video, encoding-name=H265 ! rtph265depay ! h265parse ! omxh265dec ! videoconvert ! appsink'
cap = cv2.VideoCapture(camSet)

#cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()