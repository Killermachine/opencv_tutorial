'''
Author: Killermachine
'''
import numpy as np
import cv2
import math
from canny import Canny
class HOG:
    def __init__(self):
        self.cell_matrix = None
        self.grad_x = None
        self.grad_y = None
        self.edge_magnitude = None
        self.edge_angle = None
        self.result = None
        self.canny = Canny()
        self.histogram = np.zeros(9)
        self.bins = ((-10,10),(10,30),(30,50),(50,70),(70,90),(90,110),(110,130),(130,150),(150,170))
        self.centers = (0,20,40,60,80,100,120,140,160)
        self.descriptor = None


    def genearte_gradients(self,image):
        '''
        Generating x and y gradients
        '''
        self.grad_x, self.grad_y = self.canny.gradient_generation(image)

    def genearte_magnitude(self):
        '''
        Generating edge magnitude
        '''
        self.edge_magnitude = self.canny.edge_magnitude(self.grad_x,self.grad_y)

    def genearte_edge_angle(self):
        '''
        generating edge angle
        '''
        self.edge_angle = self.canny._edge_angle(self.grad_x,self.grad_y)

    def vote_and_add(self,index1,index2,mag,angle,centers):
        '''
        Dividing the magnitude according to the angles
        '''
        bin_size = (centers[-1] - centers[0])/(len(centers)-1)
        index1_share = ((abs(angle - self.centers[index2]) )/bin_size) * mag
        index2_share = abs(mag - index1_share)
        
        return index1_share,index2_share

    def compute_cell_histogram(self, edge_magnitude, edge_angle, bins, centers):
        '''
        Computing histogram of a cell
        '''
        x,y = edge_magnitude.shape
        histogram = np.zeros(9)
        for i in range(x):
            for j in range(y):
                angle = edge_angle[i][j]
                closest_center = min(centers, key=lambda x:abs(x-angle))
                index_ = centers.index(closest_center)
                # if the angle is same as one of the centers
                if closest_center - edge_angle[i][j] == 0:
                    histogram[index_] += edge_magnitude[i][j]
                # deciding the side where the angle is wrt closest center and spliting magnitudes    
                elif closest_center - angle > 0:
                    second_index = (index_ - 1) % len(centers)
                    share1,share2 = self.vote_and_add(index_,second_index,edge_magnitude[i][j],edge_angle[i][j],centers)
                    # adding the magnitude to histogram
                    histogram[index_] += share1
                    histogram[second_index] += share2
                else:
                    second_index = (index_ + 1) % len(centers)
                    share1,share2 = self.vote_and_add(index_,second_index,edge_magnitude[i][j],edge_angle[i][j],centers)
                    # adding the magnitude to histogram
                    histogram[index_] += share1
                    histogram[second_index] += share2
        
        return histogram

    def generate_hog_features(self,image,cell_dim,block_size):
        '''
        controller function to generate hog
        '''
        # Generating magnitude and angle
        self.genearte_gradients(image)
        self.genearte_magnitude()
        self.genearte_edge_angle()

        # defining cell matrix
        x,y = self.edge_magnitude.shape
        self.cell_matrix = np.zeros((x/cell_dim[0],y/cell_dim[1],9))
        x_ = cell_dim[0]
        y_ = cell_dim[1]
        
        # generating cells
        for i in range(x/x_):
            for j in range(y/y_):
                histogram = self.compute_cell_histogram(self.edge_magnitude[i*x_: (i*x_ + x_), j*y_: (j*y_ + y_)],self.edge_angle[i*x_: (i*x_ + x_),j*y_: (j*y_ + y_)],self.bins,self.centers)
                self.cell_matrix[i][j] = histogram

        x,y,_ = self.cell_matrix.shape

        # defining descriptor
        self.descriptor = np.zeros((x-block_size/2,y-block_size/2,36))
        # Generating blocks
        for i in range(x-block_size/2):
            for j in range(y-block_size/2):
                temp = np.concatenate((self.cell_matrix[i][j],self.cell_matrix[i+1][j],self.cell_matrix[i][j+1],self.cell_matrix[i+1][j+1]), axis = None)
                # normalization
                L2_norm = math.sqrt(np.sum(temp ** 2))
                self.descriptor[i][j] = temp/L2_norm
                self.descriptor[np.isnan(self.descriptor)] = 0.0
        
        return self.descriptor  