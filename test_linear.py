from ANN import ANN 
from PSO import PSO 

#data = 'Data/1in_sine.txt'
#data = 'Data/1in_linear.txt'
data = 'Data/1in_cubic.txt'
#data = 'Data/2in_xor.txt'

input_size = 1

ann = ANN(input_size, [12,6,6,1])
pso = PSO(30, 1, 1.5, 1.5, 1, 1, ann, 100, data, input_size)

pso.execute()