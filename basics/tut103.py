# Author: killermachine

import cv2
import numpy as np
"""
This tutorial covers how to work with videos
"""
# via web cam
cap = cv2.VideoCapture(1)
while(True):
	ret,frame = cap.read()
	if not ret:
		print "Can't access"
		break
	
	cv2.imshow("video",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()

# Get video stream over IP( Android Camera )
# Pre-requisite steps
# Step 1: Install "IP Webcam" from the playstore
# Step 2: connect to the same wifi router and start server from the app
# Step 3 note the URL

# replace this with the respective IP address
url = 'http://192.168.1.102:8081/shot.jpg?rnd=737830'
import urllib
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

while(True):
	imgResp = urllib.urlopen(url)
	imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)
	frame = cv2.imdecode(imgNp,-1)

	cv2.imshow("video over IP",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.destroyAllWindows()

# Saved video
cap = cv2.VideoCapture("./data/video.mp4")
while(True):
	ret,frame = cap.read()
	if not ret:
		print "Can't access"
		break
	
	cv2.imshow("video",frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()