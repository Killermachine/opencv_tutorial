'''
Author: Killermachine
'''

import os
import numpy as np 
import cv2

class ImageLoader:
    def __init__(self):
        self.batch = []

    def load_image(self,path):
        image_list = os.listdir(path)
        for image in image_list:

            self.batch.append(cv2.imread(path + "/" + image))
        return self.batch
