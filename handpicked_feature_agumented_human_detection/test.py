from hog import HOG
import numpy as np 
import cv2

from main import Controller
from canny import Canny

positive = Controller().load_batch("./Test_positive/")
negative = Controller().load_batch("./Test_negative/")
print(len(positive))
positive = Controller().grascale(positive)
negative = Controller().grascale(negative)
print(len(positive))
i = 0
for image in positive:
    i += 1
    grad_x,grad_y = Canny().gradient_generation(image)
    egde_mag = Canny().edge_magnitude(grad_x,grad_y)
    cv2.imwrite("./results/Postive"+str(i)+".jpg",egde_mag)

i = 0
for image in negative:
    i += 1
    grad_x,grad_y = Canny().gradient_generation(image)
    egde_mag = Canny().edge_magnitude(grad_x,grad_y)
    cv2.imwrite("./results/Negative"+str(i)+".jpg",egde_mag)

image = cv2.imread("./Train_positive/crop001278a.bmp")
image = 0.114 * image[:,:,0] + 0.587 * image[:,:,1] + 0.299 * image[:,:,2]
descriptor = HOG().generate_hog_features(image,(8,8),2)
np.savetxt("crop001278a.csv", descriptor.flatten(), delimiter="\n")

image = cv2.imread("./Test_positive/crop001045b.bmp")
image = 0.114 * image[:,:,0] + 0.587 * image[:,:,1] + 0.299 * image[:,:,2]
descriptor = HOG().generate_hog_features(image,(8,8),2)
np.savetxt("crop001045b.csv", descriptor.flatten(), delimiter="\n")
