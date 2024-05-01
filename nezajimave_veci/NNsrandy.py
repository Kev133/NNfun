import random

import numpy as np
import scipy
from matplotlib import pyplot as plt
exp= np.exp
random.seed(0)

a0= 0.6 #random.randint(1,5)/10
print("Training input")
print(a0)
training_output = a0*2
#training_input = np.array([[0.1],[0.2],[0.3],[0.5]])
def sigmoid(x):
    return 1 / (1 + exp(-x))
def sigmoid_der(x):
    return x*(1-x)
def relu(x):
    """
    napise maximum, kdyz x je 5 tak 5 je vetsi nez 0 a je maximum,
    kdyz x je -2 tak 0 je vetsi nez x a je maximum
    """
    return max(0,x)

w1 = np.random.rand(3,1) #wLjk = w111,w121,w131 j=3,k=1, w131
w2 = np.random.rand(2,3) #wLjk = w223
w3 = np.random.rand(1,2) #wLjk = w312

b1 = np.array([[0,0,0]]).T
b2 = np.array([[0,0]]).T
b3= np.array([[0]])

a1 = sigmoid(np.dot(w1,a0)+b1)
a2 = sigmoid(np.dot(w2,a1)+b2)
a3 = sigmoid(np.dot(w3,a2)+b3)
print(f"training output1 is: {a3}")

error3 = (a3-training_output)*sigmoid_der(np.dot(w3,a2)+b3)
error2 = (np.dot((w3.T),error3))*sigmoid_der(np.dot(w2,a1)+b2)
error1 = (np.dot((w2.T),error2))*sigmoid_der(np.dot(w1,a0)+b1)
#print(f"error3 is {error3}")
#print(f"error2 is {error2}")
#print(f"error1 is {error1}")
cost_fun1 = np.dot(error1,a0)
cost_fun2 = np.dot(error2,a1.T)
cost_fun3 = np.dot(error3,a2.T)
#print("cost fun fun starts here")
#print(cost_fun1)
#print(cost_fun2)
#print(cost_fun3)
  #TODO update weights and biases
w1 = w1-cost_fun1
w2 = w2-cost_fun2
w3 = w3-cost_fun3
b1 = b1 -error1
b2 = b2 -error2
b3 = b3 -error3

a1 = sigmoid(np.dot(w1,a0)+b1)
a2 = sigmoid(np.dot(w2,a1)+b2)
a3 = sigmoid(np.dot(w3,a2)+b3)
print(f"training output2 is: {a3}")


