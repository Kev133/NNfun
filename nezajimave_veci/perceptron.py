import numpy as np
import matplotlib.pyplot as plt
exp = np.exp


def sigmoid(x):
    return 1 / (1 + exp(-x))
def sigmoid_der(x):
    return x*(1-x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])
training_outputs = np.array([[0, 1, 1, 0]]).T  # transposes so its a column vector

# random numbers for the initial weights, normally much more difficult choosing the initial weights
weights = np.array([[-0.2,0.4,-1]]).T

print(f"Initial synaptic weights: \n{weights}")
n = range(1000)

# syn1 = []
# syn2 = []
# syn3 = []
# outputs1=[]
# outputs2=[]
# outputs3=[]
# outputs4=[]
for number in n:

    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, weights))
    error = outputs -training_outputs 

    adjustments = error*sigmoid_der(outputs)
    synaptic_weights =  weights - np.dot(input_layer.T,adjustments)
    # syn1.append(synaptic_weights[0][0])
    # syn2.append(synaptic_weights[1][0])
    # syn3.append(synaptic_weights[2][0])
    # outputs1.append(outputs[0])
    # outputs2.append(outputs[1])
    # outputs3.append(outputs[2])
    # outputs4.append(outputs[3])

print(f"Synaptic weights after training:\n {synaptic_weights}")
print(f"Outputs after training:\n{outputs}")


# plt.plot(n, syn1)
# plt.plot(n, syn2)
# plt.plot(n, syn3)
# n4 = np.array([[n,n,n,n]]).T
# print(n4)
# plt.plot(n,outputs1,label = "x1")
# plt.plot(n,outputs2,label = "x2")
# plt.plot(n,outputs3,label = "x3")
# plt.plot(n,outputs4,label = "x4")
# plt.legend()
# plt.show()