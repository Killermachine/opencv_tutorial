#Author: killermachine

import cv2
import numpy as np 

"""
This tutorial will cover the following topics:
1. Read Image
2. Write image
3. Display image
4. Basic operations
"""

# Read and display an image

# reads an image 
img = cv2.imread("./data/sample.png")
# display: <name of the frame>,image
cv2.imshow("image",img)
# wait for key to be pressed
cv2.waitKey()
# Close all windows
cv2.destroyAllWindows() 



# Perform operations and write changes

img = cv2.imread("./data/sample.png")
# cvtColor function provides functionality to change the color scheme(grayscale in this case)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# write an image to path
cv2.imwrite("./data/gray_sample.png",gray)
cv2.imshow("gray_image",gray)
cv2.waitKey()
cv2.destroyAllWindows()



# accessing pixels, color channels, selecting ROI

img = cv2.imread("./data/sample.png")
# total number of pixels
print img.size
# shape of the pixel (row,col,color channels)
print img.shape
# access particular pixel
print img[30,40]
# set pixel value(blue,green,red)
img[30,40] = [100,100,100]
# removing red color
img[:,:,2] = 0
cv2.imshow("no red",img)
cv2.waitKey()
cv2.destroyAllWindows()
# select ROI(x:x+w,y:y+h)
ROI = img[150:500,200:550]
cv2.imshow("ROI",ROI)
cv2.waitKey()
cv2.destroyAllWindows()