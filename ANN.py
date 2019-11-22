import math
import numpy as np

##### Activation functions
def zero(x):
    return 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gaussian(x):
    return math.exp(-(x**2 / 2))

def identity(x):
    return x

def binary(x):
    if x < 0:
        return -1
    else:
        return 1

def relu(x):
    if x < 0:
        return 0
    else:
        return x

def softplus(x):
    return math.log(1 + math.exp(x))

#Dictionary used to map integer into activation functions
activation_functions_dict = {0: zero, 1: sigmoid, 2: math.tanh, 3: math.cos, 4: gaussian, 5: identity, 6:binary, 7: math.atan, 8: relu, 9: softplus}


class ANN:
    """Class implementing the neural network"""
    def __init__(self, input_size, neurons):
        self.neurons = neurons
        self.activation_functions = [np.zeros(n) for n in neurons[:-1]] #placeholder for activation functions
        self.bias = [np.zeros(n) for n in neurons] # placeholder for biases
        print(len(self.activation_functions))
        # generate placeholders for weights
        w = []
        w.append(np.zeros((neurons[0], input_size)))
        for i in range(len(neurons) - 1):
            w.append(np.zeros((neurons[i+1], neurons[i])))
        self.weights = w

    
    def calculate_net_u(self, x, k, y):
        """ Function to calculate the net u of a neuron, 
        x is the input, k is the index of the neuron in the layer, y is the number of the layer"""
        u = 0
        for j in range(len(x)):
            u = u + (self.weights[y][k][j]*x[j]) 
        return u + self.bias[y][k]
    
    def layer_output(self,x,y):
        """ Function to calculate the output of layer y that has as input x"""
        res = []
        for k in range(self.neurons[y]):
            if (y != (len(self.neurons) - 1)):
                function_index = self.activation_functions[y][k]
                f = activation_functions_dict[function_index]
                res.append(f(self.calculate_net_u(x, k, y)))
            else:
                res.append(self.calculate_net_u(x, k, y))
        return res

    def process(self, x):
        """ Function that calculates the output of the neural network given a specific input x"""
        for y in range(len(self.neurons)):
            x = self.layer_output(x, y)
        return x[0]
    
    def set_values(self, values):
        """ Function to convert the position of a particle into the parameters of the network """
        k = 0
        for y in range(len(self.weights)):
            w_old = self.weights[y]
            for i in range(len(w_old)):
                for j in range(len(w_old[i])):
                    w_old[i][j] = values[k]
                    k += 1

        for y in range(len(self.activation_functions)):
            af = self.activation_functions[y]
            for i in range(len(af)):
                af[i] = math.floor(abs(values[k]))
                k = k+1
        
        for y in range(len(self.bias)):
            b = self.bias[y] 
            for i in range(len(b)):
                b[i] = values[k]
                k = k+1

        