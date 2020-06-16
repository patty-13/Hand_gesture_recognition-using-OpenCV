#!/usr/bin/env python
# coding: utf-8

# In[1]:


import h5py
import numpy as np
import tensorflow as tf
import math

def load_dataset():
    train_dataset = h5py.File('datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3
    


# In[2]:


import math
import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
#%matplotlib inline
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
np.random.seed(1)
def linear_function():
    np.random.seed(1)
    X = tf.constant(np.random.randn(3,1), name = "X")
    W = tf.constant(np.random.randn(4,3), name = "W")
    b = tf.constant(np.random.randn(4,1), name = "b")
    Y = tf.matmul(W,X)+b    
    sess = tf.compat.v1.Session()
    #tf.Session()
    result = sess.run(Y)
    sess.close()
    return result
print( "result = \n" + str(linear_function()))

def sigmoid(z):
    x = tf.compat.v1.placeholder(tf.float32,name='x')
    sigmoid = tf.sigmoid(x)
    sess =tf.compat.v1.Session()
    result = sess.run(sigmoid,feed_dict={x:z})
    return result
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
def cost(logits, labels):    
    z =  tf.compat.v1.placeholder(tf.float32,name='z')
    y =  tf.compat.v1.placeholder(tf.float32,name='y')
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z,labels=y)
    sess = tf.compat.v1.Session()
    cost = sess.run(cost,feed_dict={z:logits,y:labels})
    sess.close()
    return cost
logits = np.array([0.2,0.4,0.7,0.9])   #for testing
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))
def one_hot_matrix(labels, C):
    C = tf.constant(C,name='C' )
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    sess = tf.compat.v1.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot
labels = np.array([1,2,3,0,2,1])       #for testing
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = \n" + str(one_hot))
def ones(shape):
    ones = tf.ones(shape)
    sess = tf.compat.v1.Session()
    ones = sess.run(ones)
    sess.close()
    return ones
print ("ones = " + str(ones([3])))
# Loading the dataset
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
print(Y_train_orig)


# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
def create_placeholders(n_x, n_y):
    X = tf.compat.v1.placeholder(tf.float32,[n_x,None],name='X')
    Y = tf.compat.v1.placeholder(tf.float32,[n_y,None],name='Y')
    return X, Y
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
def initialize_parameters():
    tf.set_random_seed(1)                  
    W1 = tf.get_variable("W1", [25,12288], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [25,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12,25], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [12,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6,12], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [6,1], initializer = tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
tf.reset_default_graph()
with tf.compat.v1.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']              
    Z1 = tf.matmul(W1,X)+b1                                              
    A1 = tf.nn.relu(Z1)                                              
    Z2 = tf.matmul(W2,A1)+b2                                              
    A2 = tf.nn.relu(Z2)                                              
    Z3 = tf.matmul(W3,A2)+b3                                              
    return Z3
tf.reset_default_graph()
with tf.compat.v1.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =logits , labels =labels ))
    return cost
tf.reset_default_graph()
with tf.compat.v1.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()                         
    tf.set_random_seed(1)                             
    seed = 3                                          
    (n_x, m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []                                        
    X, Y = create_placeholders(n_x,n_y)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer =tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    init = tf.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train 
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # ploting the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per fives)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        #saving the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        #saver = tf.train.Saver([parameters])
        #saver.save(sess, 'my_test_model',global_step=1000)

        # Calculating the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculating accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
start_time = time.time()
parameters = model(X_train, Y_train, X_test, Y_test)
print("--- %s seconds ---" % (time.time() - start_time))


# In[4]:


parameters = model(X_train, Y_train, X_test, Y_test)


# In[5]:


import scipy
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy import ndimage
import os
import cv2

currentFrame = 0
cam = cv2.VideoCapture(0)
os.makedirs('Opdtfinal4')

while(cam.isOpened()):
    ret , frame = cam.read()
    #defining region of importance
    roi = frame[100:400 , 100:400]
    # drawing rectangle on the screen
    cv2.rectangle(frame ,(100,100),(400,400),(0,255,0),0)
    # defining how gaussian blur to reduce noise
    blurred_frame = cv2.GaussianBlur(roi , (5,5) ,0)
    #converting into hsv for checking wheter the range of colour is correct
    hsv = cv2.cvtColor(blurred_frame ,cv2.COLOR_BGR2HSV)
    
    #define the hand in the bounding area or setting the range of the
    
    lower_skin = np.array([0,20,70])
    upper_skin = np.array([20,255,255])
    
    #detecting the hand in the bounding area using the skindetection
    
    mask = cv2.inRange(hsv , lower_skin ,upper_skin)
    
    #marking hand using contours
    mask,contours ,hierarchy = cv2.findContours(mask ,cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area >maxArea:
                maxArea = area
                ci = i
        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing  = np.zeros(frame.shape)
        cv2.drawContours(drawing , [res] , 0 ,(0,255,0) ,2)
        cv2.drawContours(drawing , [hull] , 0 ,(0,255,0) ,2)
        
    im = Image.fromarray(roi , 'RGB')
    im = im.resize((64,64))
    img_array = np.array(im)
    img_array = np.reshape(img_array , (1,64*64*3)).T
    my_image_prediction = predict(img_array ,parameters)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    if str(np.squeeze(my_image_prediction)) == "0":
        cv2.putText(frame ,'0',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    elif str(np.squeeze(my_image_prediction)) == "1":
        cv2.putText(frame ,'1',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    elif str(np.squeeze(my_image_prediction)) == "2":
        cv2.putText(frame ,'2',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    elif str(np.squeeze(my_image_prediction)) == "3":
        cv2.putText(frame ,'3',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    elif str(np.squeeze(my_image_prediction)) == "4":
        cv2.putText(frame ,'4',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    elif str(np.squeeze(my_image_prediction)) == "5":
        cv2.putText(frame ,'5',(0,50),font,2,(0,0,255),3,cv2.LINE_AA)
    cv2.imshow('FRAME_MAIN_CAPUTRE_AREA' , frame)
    cv2.imshow('MASK' , mask)
        
    k = cv2.waitKey(10)
        # press esc to escape from all the windows
    
    if ret == False:
        break
    elif k == 27:
        break
    
        #press b to record or capture the frames and store in file
    elif k == ord('b'):
            #for different images in folder
        name = 'frame'+ str(currentFrame)+'.jpg'
            #for overwriting the images in the folder
            #name = 'frame0.jpg'
        print("Creating..." + name)
    
        cv2.imwrite(os.path.join('OPdtfinal4',"frame{:d}.jpg".format(currentFrame)),roi)
        currentFrame +=1
cam.release()
cv2.destoryAllWindows()


# In[ ]:




