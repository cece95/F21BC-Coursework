from ANN import ANN 
from PSO import PSO 

ann = ANN(1, [8,4,1], [1,1,1])
pso = PSO(50, 0.8, 1.5, 1.5, 1, 0.5, ann, 50, 'Data/1in_linear.txt')

pso.execute()