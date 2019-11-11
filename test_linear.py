from ANN import ANN 
from PSO import PSO 

ann = ANN(1, [2,1])
pso = PSO(20, 0.8, 1.5, 1.5, 1, 1, ann, 100, 'Data/1in_linear.txt')

pso.execute()