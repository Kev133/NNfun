"""
Module for using a neural network to approximate a function between values 0 and 1 by using stochastic gradient descent.
"""
# numerical python library
import numpy as np
import matplotlib.pyplot as plt

def mini_batch_maker(batch_size):
    """
    Function for creating the training data, which in this case is a list = mini batch.
    In this list there is are pairs of x (input values) and y (output, target values).
    :param batch_size: number of x,y pairs in the mini batch
    :return: returns a list made from smaller lists made up of x,y pairs e.g. [[0.2,0.4],[0.3,0.6]]
    """

    mini_batch = []
    for num in range(0, batch_size):
        x = np.random.uniform(0,1)
        y = x**2
        mini_batch.append([x, y])

    return mini_batch


# initialize weights as random floats from -1 to 1
# the comments on the side are just for notation purposes
w1 = (np.random.randint(1, 200, (3, 1)) - 100) / 100  # wLjk = w111,w121,w131 j=3,k=1, w131
w2 = (np.random.randint(1, 200, (2, 3)) - 100) / 100  # wLjk = w223
w3 = (np.random.randint(1, 200, (1, 2)) - 100) / 100  # wLjk = w312

# initialize biases as vectors of 0s (this is recommended to do)
b1 = np.array([[0, 0, 0]]).T
b2 = np.array([[0, 0]]).T
b3 = np.array([[0]])

# other parameters for the neural network
LR = 20  # learning rate
batch_size = 5
number_of_batches = 1000


# activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of activation function
def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Main loop of the neural network
for batch in range(0, number_of_batches):
    # _sum variables, so that the cost_fun and error values can be summed in the nested for loop, they reset to 0
    # after the nested for loop ends i.e. after the last x,y pair in the mini batch is used.
    cost_fun1_sum = 0
    cost_fun2_sum = 0
    cost_fun3_sum = 0
    error1_sum = 0
    error2_sum = 0
    error3_sum = 0
    # creates a mini batch for the current loop
    mini_batch = mini_batch_maker(batch_size)
    for xy_pair in mini_batch:
        """ 
        nested for loop, this is the most important part of the program, here the activations a1,a2,a3 
        are calculated from the known initial weights and biases. This is the so-called forward pass.
        The errors and gradient of the cost function are calculated using backpropagation. The error is gradually 
        propagated backwards through the layers. The cost function gradient is calculated using these errors.
        
        """
        # unpacking the x,y pair, x is input into the "zeroth" layer of the NN, therefore it is a0
        a0 = xy_pair[0]
        y = xy_pair[1]

        # calculate the activations for the individual layers (a1=layer1)
        a1 = sigmoid(np.dot(w1, a0) + b1)
        a2 = sigmoid(np.dot(w2, a1) + b2)
        a3 = sigmoid(np.dot(w3, a2) + b3)

        # calculate the error using backpropagation for the individiual layers (error3=layer3)
        error3 = (a3 - y) * sigmoid_der(np.dot(w3, a2) + b3)
        error2 = (np.dot(w3.T, error3)) * sigmoid_der(np.dot(w2, a1) + b2)
        error1 = (np.dot(w2.T, error2)) * sigmoid_der(np.dot(w1, a0) + b1)

        # calculate the gradient of the cost function
        cost_fun1 = np.dot(error1, a0)
        cost_fun2 = np.dot(error2, a1.T)
        cost_fun3 = np.dot(error3, a2.T)

        # add the gradient to the sum variable, the sum variable sums up the gradient for the whole mini batch
        cost_fun1_sum += cost_fun1
        cost_fun2_sum += cost_fun2
        cost_fun3_sum += cost_fun3

        error1_sum += error1
        error2_sum += error2
        error2_sum += error2

    # adjust the weights and biases using the average gradient of the cost function and average errors
    w1 = w1 - LR * cost_fun1_sum / batch_size
    w2 = w2 - LR * cost_fun2_sum / batch_size
    w3 = w3 - LR * cost_fun3_sum / batch_size
    b1 = b1 - LR * error1_sum / batch_size
    b2 = b2 - LR * error2_sum / batch_size
    b3 = b3 - LR * error3_sum / batch_size


# testing the neural network on data that it has not seen
test_batch = mini_batch_maker(15)
input_list = []
output_list = []
target_output_list = []
for xy in test_batch:
    test_input = xy[0]
    test_output = xy[1]
    a1 = sigmoid(np.dot(w1, test_input) + b1)
    a2 = sigmoid(np.dot(w2, a1) + b2)
    a3 = sigmoid(np.dot(w3, a2) + b3)

    input_list.append(test_input)
    output_list.append(a3[0][0])
    target_output_list.append(test_output)

#print(f"input is {input_list}")
#print(f"output is: {output_list}")
xx = np.linspace(0, 1, 100)  # 100 points between 0 and 1
yy_square = xx ** 2
mock_y_sin = np.sin(xx)

# Create the plot
plt.plot(xx, yy_square, label='y = x^2', color='r')
#plt.plot(mock_x, mock_y_sin, label='y = sinx', color='g')
plt.plot(input_list,output_list,"bo",label ="NN_test_results")
plt.legend()

plt.show()