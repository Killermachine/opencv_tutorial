#Author: killermachine

import cv2
import numpy as np 

"""
This tutorial will cover drawing on images
"""
# Line

img = cv2.imread("./data/sample.png")
# Drawing a line(x1,y1),(x2,y2),(color),width
cv2.line(img,(10,10),(200,500),(0,0,255),5)
cv2.imshow("Draw Line",img)
cv2.waitKey()
cv2.destroyAllWindows()

# Rectangle

img = cv2.imread("./data/sample.png")
# Drawing a rectangle(x1,y1),(x2,y2),(color),width
cv2.rectangle(img,(300,50),(500,500),(0,0,255),3)
cv2.imshow("Draw rectangle",img)
cv2.waitKey()
cv2.destroyAllWindows()

# circle

img = cv2.imread("./data/sample.png")
# Drawing a circle(x1,y1),radius,(color),width
cv2.circle(img,(300,50),100,(0,0,255),3)
cv2.imshow("Draw circle",img)
cv2.waitKey()
cv2.destroyAllWindows()

# polygon

img = cv2.imread("./data/sample.png")
# Drawing a polygon
points = np.array([[100,50],[200,400],[700,200],[500,1000]], np.int32)
points = points.reshape((-1,1,2))
cv2.polylines(img,[points],True,(0,0,255))
cv2.imshow("Draw polylines",img)
cv2.waitKey()
cv2.destroyAllWindows()

# Text

img = cv2.imread("./data/sample.png")
# image, (x,y),font,scaling, thinkness and type
cv2.putText(img, "Hello World",(100,500),cv2.FONT_HERSHEY_SIMPLEX,4, (0,0,255),2,cv2.LINE_AA)
cv2.circle(img,(100,500),5,(0,0,255),3)
cv2.imshow("Write on image",img)
cv2.waitKey()
cv2.destroyAllWindows()
