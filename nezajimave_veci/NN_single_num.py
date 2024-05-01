import random
import numpy as np
import scipy
from matplotlib import pyplot as plt
exp= np.exp
random.seed(3)
"""
Module for using a neural network to approximate a function by regression using stochastic gradient descent.
In czech = stochastický gradientní sestup
"""
def sigmoid(x):
    return 1 / (1 + exp(-x))
def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))
def relu(x):
    """
    napise maximum, kdyz x je 5 tak 5 je vetsi nez 0 a je maximum,
    kdyz x je -2 tak 0 je vetsi nez x a je maximum
    """
    return max(0,x)


# Initialize weights with Xavier initialization
# w1 = np.random.randn(3, 1) * np.sqrt(2 / (3 + 1))
# w2 = np.random.randn(2, 3) * np.sqrt(2 / (2 + 3))
# w3 = np.random.randn(1, 2) * np.sqrt(2 / (1 + 2))
#initialize weights as numbers and biases as vectors of 0s
w1 = (np.random.randint(1,100,(3,1))-50)/100 #wLjk = w111,w121,w131 j=3,k=1, w131
w2 = (np.random.randint(1,100,(2,3))-50)/100 #wLjk = w223
w3 = (np.random.randint(1,100,(1,2))-50)/100 #wLjk = w312

b1 = np.array([[0,0,0]]).T
b2 = np.array([[0,0]]).T
b3= np.array([[0]])


#a0 = random.randint(1,5)/10
a0= 0.2
y = 2*a0
LR = 1
for number in range (0,30):
    # calculate the activations for the individual layers
    a1 = sigmoid(np.dot(w1,a0)+b1)
    a2 = sigmoid(np.dot(w2,a1)+b2)
    a3 = sigmoid(np.dot(w3,a2)+b3)
    ss = (a3-y)**2
    if number ==0:
        print(f"first training output is: {a3}")

    error3 = (a3-y)*sigmoid_der(np.dot(w3,a2)+b3)  # suddenly works, when I use sigmoid instead of sigmoid_der
    error2 = (np.dot(w3.T,error3))*sigmoid_der(np.dot(w2,a1)+b2)
    error1 = (np.dot(w2.T,error2))*sigmoid_der(np.dot(w1,a0)+b1)

    cost_fun1 = np.dot(error1,a0)
    cost_fun2 = np.dot(error2,a1.T)
    cost_fun3 = np.dot(error3,a2.T)
    w1 = w1-LR*cost_fun1
    w2 = w2-LR*cost_fun2
    w3 = w3-LR*cost_fun3
    b1 = b1 -LR*error1
    b2 = b2 -LR*error2
    b3 = b3 -LR*error3
    #print(f"output from iteration {number} is {a3[0][0]}")
print(f"input is {a0}")
print(f"output is: {a3}")
# a0 = 0.1
# a1 = sigmoid(np.dot(w1,a0)+b1)
# a2 = sigmoid(np.dot(w2,a1)+b2)
# a3 = sigmoid(np.dot(w3,a2)+b3)
# print(f"test input is {a0}")
# print(f"test output is: {a3}")

mF=500 #kg/h
kumul= np.array([100,85,24,10])/100 #%
L = np.array([[1000,750,500,250]])

print(L)
hmot = [10,24-10,85-24,100-85]




















