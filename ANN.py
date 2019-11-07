import math

def zero(x):
    return 0

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def gaussian(x):
    return math.exp(-(x**2 / 2))

activation_functions_dict = {0: zero, 1: sigmoid, 2: math.tanh, 3: math.cos, 4: gaussian}

x = [0,1]

class ANN:
    def __init__(self, input_size, neurons, activation_functions):
        self.neurons = neurons
        self.activation_functions = activation_functions
        if len(neurons) != len(activation_functions):
            print("Error in ANN initialization, the number of layers and the number of activation functions should be the same")

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
        self.weights = weights

ann = ANN(len(x), [2,2,1], [1,1,1])