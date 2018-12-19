'''
@Author: Tanay Varshney (tv655)
'''

import numpy as np

class NeuralNetwork:
  def __init__(self, num_layers, layer_dimensions, drop_prob, reg_lambda = 0):
    
    self.reg = reg_lambda
    self.num_layers = 1 + len(layer_dimensions)
    self.params = {}
    self.weight_scale = 2.461858e-02 
    self.fc_cache = {}
    self.relu_cache = {}
    self.batch_size = 20
    
    input_dim = 19*11*36
    
    for i in range(self.num_layers - 1):
      self.params['W' + str(i+1)] = np.random.normal(0, self.weight_scale, [input_dim, layer_dimensions[i]])
      self.params['b' + str(i+1)] = np.zeros([layer_dimensions[i]])
      input_dim = layer_dimensions[i]  

    self.params['W' + str(self.num_layers)] = np.random.normal(0, self.weight_scale, [input_dim, 2])
    self.params['b' + str(self.num_layers)] = np.zeros([2])

  
  def relu (self, x):
    '''
    Relu activation function
    :param x: input value
    '''
    relu_func = lambda x: max(0,x)
    return relu_func
  
  def sigmod(self,x):
    '''
    Sigmod activation function
    '''
    return 1/(1+np.exp(-x))
  
  def back_sigmoid(self,x):
    '''
    Gradients of sigmoid in backpass
    '''
    return x*(1-x)

  def affineForward(self, A, W, b):
    '''
    The affine forward pass
    @param x: input matrix
    @param w: weight matrix
    @param b: bias matrix
    '''
    out = None
    NN = A.shape[0]
    reshaped_input = np.reshape(A, [NN, -1])
    out = np.dot(reshaped_input, W) + b
    cache = (A, W, b)
    return out, cache
  
  def activationForward(self,x):
    '''
    This function applies relu activation function
    @param x: Input
    '''
    out = None
    out = x.copy()
    out[out < 0] = 0
    cache = x
    return out, cache

  def dropout(self, A, prob):
    '''
    @param A: Input Matrix
    @param prob: A dropout probabilty percentage
    '''
    mask = (np.random.rand(*A.shape) < prob) / prob
    out = A * mask
    return out, mask

  def forwardPropogation(self, X):
    '''
    This function ties together the forward pass
    @param X: input image
    '''
    size = X.shape[0]
    X = np.reshape(X, [size, -1])  
    
    for i in range(self.num_layers-1):
        fc_act, self.fc_cache[str(i+1)] = self.affineForward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        relu_act, self.relu_cache[str(i+1)] = self.activationForward(fc_act)
        X = relu_act.copy()
        
    scores, final_cache = self.affineForward(X, self.params['W'+str(self.num_layers)], self.params['b'+str(self.num_layers)])  
    return scores, final_cache
  
  
  def affineBackward (self, dAl, cache):
    '''
    This function performs the backward pass on the fully connected layers
    @param dAl: Gradient
    @param cache: The cached parameters from the forward pass
    '''
    x, w, b = cache
    dx, dw, db = None, None, None
    NN = x.shape[0]
    reshaped_x = np.reshape(x,[NN, -1])
    dx = np.dot(dAl, w.T)
    dx = np.reshape(dx, x.shape)
    dw = np.dot(reshaped_x.T,dAl)
    db = np.sum(dAl, axis=0)
    return dx, dw, db
  

  def activationBackward(self, dout, cache):
    '''
    This function is used to perfrom backward pass through the relu. Returns the gradient
    '''
    dx, x = None, cache
    relu_mask = (x >= 0)
    dx = dout * relu_mask
    return dx

  
  
  def backPropogation(self, scores, y, final_cache):
    '''
    This function basically controls the backward pass in through the layers
    @param scores: Result of the forward pass 
    @param y: Real classes
    @param final_cache: Cache to keep track of features and weights
    '''
    loss, grads = 0.0, {}
    loss, dsoft = self.softmax_loss(scores, y)
    loss += 0.5*self.reg*(np.sum(np.square(self.params['W'+str(self.num_layers)])))
    
    dx_last, dw_last, db_last = self.affineBackward(dsoft, final_cache)
    
    grads['W'+str(self.num_layers)] = dw_last + self.reg*self.params['W'+str(self.num_layers)]
    grads['b'+str(self.num_layers)] = db_last

    for i in range(self.num_layers-1, 0, -1):
        drelu = self.activationBackward(dx_last, self.relu_cache[str(i)])
        dx_last, dw_last, db_last = self.affineBackward(drelu, self.fc_cache[str(i)])
        grads['W' + str(i)] = dw_last + self.reg * self.params['W' + str(i)]
        grads['b' + str(i)] = db_last
        loss += 0.5 * self.reg * (np.sum(np.square(self.params['W' + str(i)])))

    return loss, grads    
    
  
  def softmax_loss(self, x, y):
    '''
    This function implements the softmax function
    @param x: Inputs
    @param y: Output class
    '''
    shiftedLogits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shiftedLogits), axis=1, keepdims=True)
    log_probs = shiftedLogits - np.log(Z)
    
    probs = np.exp(log_probs) 
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()

    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

  
  def train(self, X, y, print_every=20, num_epochs=20, batch_size=20, alpha = 0.15):
    '''
    This is the training function where the main loop to train the designed architecture recides
    @param X: The training set
    @param y: The classes
    @param print_every: Determines the number of iterations after which feedback is provided
    @param num_epochs: The total number of epochs
    @param batch_size: The batch size in use during the training process
    @param alpha: The learning parameter
    '''
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size , 1)
    
    num_iterations= 200
    print("Total iterations:"+ str(num_iterations))

    for i in range(num_iterations):

      num_train = X.shape[0]
      batch_mask = np.random.choice(num_train, self.batch_size)
      X_batch = X[batch_mask]
      y_batch = y[batch_mask]
      scores, final_cache = self.forwardPropogation(X_batch)
      loss, grads = self.backPropogation(scores, y_batch, final_cache)
      self.updateParameters(grads,alpha)
      
      if i % print_every == 0:
        print(str(i)+"/"+str(num_iterations)+" iterations done. At i="+str(i)+" => | Loss: "+str(loss) + " | Accuracy:"+str(self.accuracy(np.argmax(scores,axis=1),y_batch)))
        
    print("Done")
      
  def updateParameters(self,gradients,alpha):
    '''
    This function updates the gradients obtained using gradient descent
    @param gradients: Computed gradients through backpropogation
    @param alpha: The learning rate for gradient descent
    '''
    for p, w in self.params.items():
      dw = gradients[p]
      prev_dw = dw
      self.params[p] = self.params[p] - dw * alpha #alpha*dw
           
  def predict(self,X_test):
    '''
    This function is used to test the trined classifier by performing predictions on the test set
    @param X_test: The test set
    '''
    y_pred = []
    scores, cache = self.forwardPropogation(X_test)    
    y_pred.append(np.argmax(scores, axis=1))                 
    return y_pred
  
  def accuracy(self,y_pred,y_test):
    '''
    This function tests the accuracy of the trained model
    @param y_pred: Predicted class 
    @param y_test: Actual Class
    '''
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y_test)
    return acc

"""## Training Time"""


def save_parameters(filename,y):
  '''
  This function is used to save the parameter
  @param filename: Name of the file where classes are stored
  @param y: The predicted values of y
  '''
  np.save(filename, y)

#save_parameters('ans1.npy', np.hstack(y_pred))

