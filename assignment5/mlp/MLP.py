import tensorflow as tf
from tensorflow.keras import initializers
import numpy as np
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from copy import deepcopy

class MLP:
    def __init__(self):
        self.lr = 5e-4
        self.input_size = 4
        self.hidden_size = 3
        self.output_size = 3
        self.hist = {'loss':[], 'acc':[]}  
        #####  initialize weights ############ 
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros(self.output_size)
        
    def softmax(self, x):
        #### to avoid overflow substract by max ####
        exps = np.exp( x - np.max(x, axis = 1, keepdims = True))
        return exps / (np.sum(exps, axis=1, keepdims=True))
        pass
    
    def sigmoid(self, x):
        return tf.math.sigmoid(x).numpy()
        pass

    def cross_entropy(self, y, o):
        n_samples = y.shape[0]
        # to avoid zero in log , we add an small number to input ####
        loss = -np.sum(np.log(o[range(n_samples),y] + 1e-5))
        return loss
        pass
    
    def forward(self, x):
        h1 = self.sigmoid(x @ self.W1+ self.b1)
        o = self.softmax(h1 @ self.W2+ self.b2)
        return o, h1
        pass

    def backward(self, y, o, h1, X):
        n_samples = y.shape[0]
        #### back propag using chainrule #####
        dlo = deepcopy(o)
        dlo[np.arange(n_samples), y] -= 1
        dlW2 = np.transpose(h1) @ dlo
        dlb2 = np.sum(dlo, axis=0)
        dlh1 = dlo @ np.transpose(self.W2)
        dlsigmoid = h1 * (1 - h1)
        dlW1 = np.transpose(X) @ dlsigmoid
        dlb1 = np.sum(dlsigmoid, axis=0)

        #### update weights ####
        self.W1 -= self.lr * dlW1
        self.b1 -= self.lr * dlb1
        self.W2 -= self.lr * dlW2
        self.b2 -= self.lr * dlb2
    

    
    def train(self, x, y, epochs):
        for epoch in tqdm(range(1, epochs+1)):
            o, h1 = self.forward(x)
            loss = self.cross_entropy(y, o)
            self.backward(y, o, h1, x)
            acc = accuracy_score(y, np.argmax(o, axis=1))
            self.hist['loss'] += [loss]
            self.hist['acc'] += [acc]
            print(epoch, 'loss:', loss, 'acc:', acc)