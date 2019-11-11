import math
import numpy as np

def zero(x):
    return 0

def sigmoid(x):
    #print(x)
    return 1 / (1 + math.exp(-x))

def gaussian(x):
    return math.exp(-(x**2 / 2))

activation_functions_dict = {0: zero, 1: sigmoid, 2: math.tanh, 3: math.cos, 4: gaussian}

class ANN:
    def __init__(self, input_size, neurons, activation_functions):
        self.neurons = neurons
        self.activation_functions = activation_functions
        if len(neurons) != len(activation_functions):
            print("Error in ANN initialization, the number of layers and the number of activation functions should be the same")
        w = []
        w.append(np.zeros((neurons[0], input_size)))
        for i in range(len(neurons) - 1):
            w.append(np.zeros((neurons[i+1], neurons[i])))
        self.weights = w

    def calculate_net_u(self, x, k, y): #x is the input is the index of the neuron in the layer, y is the number of the layer
        u = 0
        for j in range(len(x)):
            u = u + (self.weights[y][k][j]*x[j]) 
        return u
    
    def layer_output(self,x,y):
        function_index = self.activation_functions[y]
        f = activation_functions_dict[function_index]
        return [f(self.calculate_net_u(x, k, y)) for k in range(self.neurons[y])]

    def process(self, x):
        for y in range(len(self.neurons)):
            x = self.layer_output(x, y)
        return x[0]
    
    def set_weights(self, weights):
        k = 0
        for y in range(len(self.weights)):
            w_old = self.weights[y]
            for i in range(len(w_old)):
                for j in range(len(w_old[i])):
                    w_old[i][j] = weights[k]
                    k += 1