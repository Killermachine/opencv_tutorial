'''
Author: Killermachine
'''

import numpy as np
import cv2
from ImageLoader import ImageLoader
from fcn import NeuralNetwork

class Controller:
    def __init__(self):
        from hog import HOG
        self.hog_generator = HOG()

    def load_batch(self,path):
        '''
        Loading a image batch
        '''
        batch_loader = ImageLoader()
        
        return batch_loader.load_image(path)

    def grascale(self,image_batch):
        '''
        images are loaded bgr
        convert from rgb to gray
        '''
        new_batch = []
        for image in image_batch:
            frame = 0.114 * image[:,:,0] + 0.587 * image[:,:,1] + 0.299 * image[:,:,2]
            new_batch.append(frame)

        return new_batch

    def genearte_hog(self,image_batch):
        '''
        Hook to generate hog, written to handle batches
        '''

        batch = []
        
        for image in image_batch:
            descriptor = self.hog_generator.generate_hog_features(image,(8,8),2)
            batch.append(descriptor)
        
        return batch

    def randomize_input_batch(self,batch1,batch2):
        '''
        Randomize dataset for training
        '''
        import random
        batch = []
        label = []
        for _ in range(len(batch1 + batch2)):
            # randomly picking a positive or a negative sample
            r = random.randint(0,1)
            if (r == 1):
                try:
                    batch.append(batch1[0])
                    label.append(1)
                    batch1 = batch1[1:]
                except:
                    batch.append(batch2[0])
                    label.append(0)
                    batch2 = batch2[1:]
            if (r == 0):
                try:
                    batch.append(batch2[0])
                    label.append(0)
                    batch2 = batch2[1:]
                except:
                    batch.append(batch1[0])
                    label.append(1)
                    batch1 = batch1[1:]
        return batch,label



    def run(self):
        '''
        Main running function

        '''
        # loading image batch
        print("Loading image")
        train_positive = self.load_batch("./Train_positive/")
        train_negative = self.load_batch("./Train_negative/")
        test_positive = self.load_batch("./Test_positive/")
        test_negative = self.load_batch("./Test_negative/")
        print(len(test_positive))
        print("Image Batch Loaded")

        # Generating grascale images
        print("Converting to grayscale")
        train_positive_gray = self.grascale(train_positive)
        train_negative_gray = self.grascale(train_negative)
        test_positive_gray = self.grascale(test_positive)
        test_negative_gray = self.grascale(test_negative)
        print("Grayscale computed")

        print("Extracting HOG descriptor")
        # Extracting Hog features for training images 
        train_positive_descriptor = self.genearte_hog(train_positive_gray)
        train_negative_descriptor = self.genearte_hog(train_negative_gray)
        
        test_positive_descriptor = self.genearte_hog(test_positive_gray)
        test_negative_descriptor = self.genearte_hog(test_negative_gray)
        print("Extracted descriptor")
        # training a neural net
        for num_neuron in [250,500,1000]:
            nn = NeuralNetwork(1, [num_neuron], 0.05)
            
            X_, Y_ = self.randomize_input_batch(train_positive_descriptor,train_negative_descriptor)
            X_train = np.array(X_, dtype=np.float32)
            
            y_train = np.array(Y_, dtype=np.int32)
            nn.train(X_train,y_train)

            # testing the network
            x_, y_ = self.randomize_input_batch(test_positive_descriptor, test_negative_descriptor)
            X_test = np.array(test_positive_descriptor + test_negative_descriptor, dtype=np.float32)
            Y_test = np.array(len(test_positive_descriptor) * [1] + len(test_negative_descriptor) * [0], dtype=np.int32)
            y_pred = nn.predict(X_test)

            """## Results and final comment"""

            print(nn.accuracy(y_pred,Y_test))
            print("->>>>>>>>>>>>>>>>>>>")
            print(y_pred)
            print("-..................")
            print(Y_test)
            

controller = Controller()
controller.run()