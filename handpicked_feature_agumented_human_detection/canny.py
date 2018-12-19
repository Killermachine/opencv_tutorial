'''
@Author: Tanay Varshney
@NetID: tv655

How to run the code:
Put the input image in the same folder as this code

Use the command

python canny.py <filename>

'''

import cv2
import numpy as np 
import math

class Canny:
    '''
    This is the main class used to generate canny edges
    '''
    def __init__(self):
        '''
        We define the basic initial settings to build a canny model
        '''
        self.image = None
        self.gausian_mask = np.array([ [1, 1, 2, 2, 2, 1, 1],
                                        [1, 2, 2, 4, 2, 2, 1], 
                                        [2, 2, 4, 8, 4, 2, 2],
                                        [2, 4, 8, 16, 8, 4, 2],
                                        [2, 2, 4, 8, 4, 2, 2],
                                        [1, 2, 2, 4, 2, 2, 1],
                                        [1, 1, 2, 2, 2, 1, 1]
                                        ],
                                        np.int64)
        self.prewitt_mask_y = np.array([ [1.0, 1.0, 1.0], [0.0, 0.0, 0.0] , [-1.0, -1.0, -1.0] ], np.int64)
        self.prewitt_mask_x = np.array([ [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0] , [-1.0, 0.0, 1.0] ], np.int64)

    def _load_image(self,path):
        '''
        Helper function: Loading image
        @param path: path( os path) where image needs to be read from
        '''
        self.image = cv2.imread(path)
        self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
    
    def _write_image(self,name,image):
        '''
        Helper function: Write image
        @param name: name of the image file
        @param image: The image that needs to be saved
        '''
        cv2.imwrite("./" + name + ".bmp",image)

    def _multiply_mat_and_sum(self,mat1,mat2,normalization_factor):
        '''
        helper function for multiply 
        @param mat1: One of the matrices involved in the multiplication
        @param mat2: One of the matrices involved in the multiplication
        @param normalization_factor: Factor used to divide the resultant multiplication of matrices in order to keep pixel lower than 255
        '''
        sum = 0
        x,y = mat1.shape
        out = np.zeros((x,y))
        for i in range(x):
            for j in range(y):
                out[i][j] = mat1[i][j] * mat2[i][j]
                sum += mat1[i][j] * mat2[i][j]

        return sum/normalization_factor

    def _convolution_function(self,mask_dim,mask,normalization_factor,image):
        '''
        Helper function: Implementing Convolution over the image
        @param mask_dim: Value of 'n' for the n x n mask to be applied
        @param mask: The mask which is being used to perform convolution
        @param normalization_factor: Factor used to divide the resultant multiplication of matrices in order to keep pixel lower than 255
        @param image: The image which needs to be convolved over
        '''
        out = np.zeros(image.copy().shape)
        x,y = image.shape
        pad_ = int(math.floor(mask_dim/2))
        
        # performing convolution    
        for i in range(pad_ ,x - pad_):
            for j in range(pad_ ,y - pad_):

                temp = self._multiply_mat_and_sum(mask,image[i - pad_ : i + pad_ + 1, j - pad_ : j + pad_ + 1],normalization_factor)
                out[i][j] = temp

        return out

    def gausian_smoothening(self,image):
        '''
        Status: Completed
        Function for gausian smoothening
        : (1) Normalized image result after Gaussian smoothing.

        @param image: The image that needs to be smoothened
        '''
        print("Applying Gausian Smoothening")
        image_after_smoothening = self._convolution_function(7,self.gausian_mask,140.0,image)
        self._write_image("smooth",image_after_smoothening)

        print("Applied Gausian Smoothening")
        return image_after_smoothening

    def gradient_generation(self,image):
        '''
        Status: Completed
        Generate gradient for canny
        : (2) Normalized horizontal and vertical gradient responses

        @param image: The image from which gradients need to be generated
        '''
        #print("Generating Gradients")
        gradient_x = self._convolution_function(3,self.prewitt_mask_x,3.0,image)      
        gradient_y = self._convolution_function(3,self.prewitt_mask_y,3.0,image)
        
        #self._write_image("gradient_x", abs(gradient_x))
        #self._write_image("gradient_y", abs(gradient_y))

        #print("gradients generated")
        return abs(gradient_x), abs(gradient_y)
    
    def edge_magnitude(self,grad_x,grad_y):
        '''
        Status: Completed
        Calculating the gradient from the vertical and the horizontal edges
        : (3) Normalized edge magnitude image

        @param grad_x: X gradient of image after applying prewitts
        @param grad_y: Y gradient of image after applying prewitts
        '''
        #print("calculating edge magnitude")
        edge_magnitude = np.sqrt(np.add( grad_x**2, grad_y**2))/math.sqrt(2)
        
        self._write_image("edge magnidute", edge_magnitude)
        return edge_magnitude

    def _edge_angle(self,grad_x,grad_y):
        '''
        Helper function to generate edge angles

        @param grad_x: X gradient of image after applying prewitts
        @param grad_y: Y gradient of image after applying prewitts
        '''
        x,y = grad_x.shape
        edge_angle = np.zeros((x,y), np.float32)
        
        # calculating edge angle
        for i in range(x):
            for j in range(y):
                if grad_x[i][j] != 0:
                    edge_angle[i][j] = math.degrees(math.atan2(grad_y[i][j],grad_x[i][j] ))
                else:
                    if (grad_y[i][j]<0):
                        edge_angle[i][j]=-90
                    elif (grad_y[i][j]>0):
                        edge_angle[i][j]=90
                    else:
                        edge_angle[i][j]=0
                # Modification for hog 
                if(edge_angle[i][j]<-10):
                    edge_angle[i][j]+=360
                if edge_angle[i][j]>=170:
                    edge_angle[i][j]=edge_angle[i][j]-180
        return edge_angle

    def non_maxima_suppression(self,edge_magnitude,edge_angle):
        '''
        Status: Completed
        Applying non maxima suppression
        : (4) Normalized edge magnitude image after non-maxima suppression.

        @param edge_magnitude: The normailized magnitude of combined x and y gradients
        @param edge_angle: The angle of combined x and y gradients
        '''
        
        x,y = edge_magnitude.shape


        # Setting up that 
        for i in range(1,x-1):
            for j in range(1,y-1):
                if edge_angle[i][j] >= -22.5  and edge_angle[i][j] < 22.5:
                    # 0 horizontal
                    if max(edge_magnitude[i][j],edge_magnitude[i][j+1],edge_magnitude[i][j-1]) != edge_magnitude[i][j]:
                        edge_magnitude[i][j] = 0
                elif edge_angle[i][j] >= 22.5  and edge_angle[i][j] < 67.5:
                    # 1 
                    if max(edge_magnitude[i][j],edge_magnitude[i-1][j+1],edge_magnitude[i+1][j-1]) != edge_magnitude[i][j]:
                        edge_magnitude[i][j] = 0
                elif edge_angle[i][j] >= 67.5  and edge_angle[i][j] < 90:
                    # 2 
                    if max(edge_magnitude[i][j],edge_magnitude[i-1][j],edge_magnitude[i+1][j]) != edge_magnitude[i][j]:
                        edge_magnitude[i][j] = 0
                elif edge_angle[i][j] >= -90  and edge_angle[i][j] < -67.5:
                    # 2
                    if max(edge_magnitude[i][j],edge_magnitude[i-1][j],edge_magnitude[i+1][j]) != edge_magnitude[i][j]:
                        edge_magnitude[i][j] = 0
                elif edge_angle[i][j] >= -67.5  and edge_angle[i][j] < -22.5:
                    # 3
                    if max(edge_magnitude[i][j],edge_magnitude[i-1][j-1],edge_magnitude[i+1][j+1]) != edge_magnitude[i][j]:
                        edge_magnitude[i][j] = 0
        self._write_image("after_suppression",edge_magnitude)
        return edge_magnitude


        

    def p_tile_thresholding(self,image,p):
        '''
        Status: Completed
        p tile thrsholding
        : (5) Generate a set of binary edge images by thresholding the normalized edge magnitude image after non-maxima suppression
        
        @param image: The image that needs to be thresholded
        @param p: The percentage in the "p-tile" method for thresholding
        '''
        x,y = image.shape
        out = image.copy()
        num_edge_pixels = 0
        bitmap = []
        

        # figuring out number of non zero pixels
        for i in range(x):
            for j in range(y):
                if i not in range(3,x-3) and j not in range(3, y-3):
                    out[i][j] = 0
                if out[i][j] > 0:
                    bitmap.append(out[i][j])
        bitmap = sorted(bitmap,reverse = True)

        # figuring out the threshold value
        threshold = int(len(bitmap) * p /100)  
        
        print("Threshold for "+ str(p) +"% is " + str(bitmap[threshold]))
        
        # applying the threshold and calcuating number of edge pixels
        for i in range(x):
            for j in range(y):
                if out[i][j] > bitmap[threshold]:
                    out[i][j] = 255
                    num_edge_pixels += 1
                else:
                    out[i][j] = 0
        print(len(bitmap))
        print("Number of edges/ Number of edge pixels = " + str(num_edge_pixels))
        print("\n")
        self._write_image("final_image"+str(p)+"%",out)
        return out

