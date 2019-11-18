from ANN import ANN 
from PSO import PSO
import time

#data = 'Data/1in_linear.txt'
#data = 'Data/1in_cubic.txt'
data = 'Data/1in_sine.txt'
#data = 'Data/1in_tanh.txt'
#data = 'Data/2in_xor.txt'

input_size = 1

#alpha (inertia weight) between 0.4 and 0.9 https://www.researchgate.net/post/What_are_the_best_PSO_parameter_values

ann = ANN(input_size, [12,1])
pso = PSO(100, 10, 0.4, 3, 1, 0, 1, ann, 1000, data, input_size)

start = time.time()
pso.execute()
print('Execution time: {} minutes'.format((time.time() - start)/60))