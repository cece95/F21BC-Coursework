import numpy.random as rand
import math

def zero(x):
    return 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gaussian(x):
    return math.exp(-(x**2 / 2))

activation_functions_dict = {0: zero, 1: sigmoid, 2: math.tanh, 3: math.cos, 4: gaussian}

x = [0,1,0,1]

class ANN :
    def __init__(self, n_neurons, n_layers, input_size, activation_functions):
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.activation_functions = activation_functions
        self.weights = 2 * rand.random_sample((n_neurons, input_size, n_layers)) - 1
        self.output_weights = 2 * rand.random_sample((1, n_neurons)) - 1 
        print(self.weights.shape)

    def calculate_net_u(self, x, k, y): #k is the index of the neuron in the layer, y is the number of the layer
        u = 0
        for j in range(len(x)):
            u = u + (self.weights[k][j][y]*x[j]) 
        return u

    def layer_output(self,x,y):
        function_index = self.activation_functions[y]
        f = activation_functions_dict[function_index]
        return [f(self.calculate_net_u(x, k, y)) for k in range(self.n_neurons)]
    
    def last_layer_output(self, x):
        function_index = self.activation_functions[-1]
        f = activation_functions_dict[function_index]
        u = 0
        for j in range(len(x)):
            u = u + (self.output_weights[j]*x[j]) 
        return f(u)

    def process(self, x):
        for y in range(self.n_layers):
            x = self.layer_output(x, y)
        return self.last_layer_output(x)

ann = ANN(2,2,len(x), [1,1,1])
print(ann.weights)
print(ann.process(x))


#class PSO :
 #   def __init__(self, )