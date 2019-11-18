import math
import numpy as np

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

activation_functions_dict = {0: zero, 1: sigmoid, 2: math.tanh, 3: math.cos, 4: gaussian, 5: identity, 6:binary, 7: math.atan, 8: relu, 9: softplus}

class ANN:
    def __init__(self, input_size, neurons):
        self.neurons = neurons
        self.activation_functions = [np.zeros(n) for n in neurons]
        self.bias = np.zeros(len(neurons))
        if len(neurons) != len(self.activation_functions):
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
        return u + self.bias[y]
    
    def layer_output(self,x,y):
        res = []
        for k in range(self.neurons[y]):
            function_index = self.activation_functions[y][k]
            f = activation_functions_dict[function_index]
            res.append(f(self.calculate_net_u(x, k, y)))
        return res

    def process(self, x):
        for y in range(len(self.neurons)):
            x = self.layer_output(x, y)
        return x[0]
    
    def set_values(self, values):
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
            self.bias[y] = values[k]
            k+1

        