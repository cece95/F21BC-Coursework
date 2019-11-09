import numpy.random as rand
import numpy as np
import pandas as pd  
import random
import math
import matplotlib.pyplot as plt

#Generate random particle to init PSO algorithm
def generate_random_particle(_id, input_size, neurons):
    position = []
    speed = []
    first_layer = 2 * rand.random_sample((neurons[0], input_size)) - 1
    first_layer_speed = np.zeros((neurons[0], input_size)) 
    position.append(first_layer)
    speed.append(first_layer_speed)
    for i in range(len(neurons) - 1):
        layer = 2 * rand.random_sample((neurons[i+1], neurons[i])) - 1
        position.append(layer)
        s = np.zeros((neurons[i+1], neurons[i]))
        speed.append(s)
    return Particle(_id, position, speed)

#Class to represent the Particle
class Particle:
    def __init__(self, id_, position, speed):
        self.id = id_
        self.position = position
        self.speed = speed
        self.fitness = 0
        self.best_fitness = math.inf 
        self.best_fitness_position = None
        self.informants = []

    # Function to randomly assign a set of informants to the particle
    def select_informants(self, swarm, n_informants):
        data = [p for p in range(0, len(swarm))]
        data.remove(self.id)
        #print(data)
        random.shuffle(data)
        for i in data[:n_informants]:
            self.informants.append(swarm[i])
    
    # Function to get the previous fittest position among the informants    
    def get_previous_fittest_of_informants(self):
        fittest_p = self.informants[0]
        for p in self.informants:
            if p.fitness < fittest_p.fitness:
                fittest_p = p
        return fittest_p.position

    def update_position(self, epsilon):
        for i in range(len(self.position)):
            for j in range(len(self.position[i])):
                for k in range(len(self.position[i][j])):
                    tmp_pos = self.position[i][j][k] + epsilon * self.speed[i][j][k]
                    # bouncing
                    while tmp_pos < -1 or tmp_pos > 1:
                        if tmp_pos > 1:
                            tmp_pos = 2 - tmp_pos
                        elif tmp_pos < -1:
                            tmp_pos = -2 - tmp_pos
                    
                    self.position[i][j][k] = tmp_pos 


class PSO:
    def __init__(self, swarm_size, alpha, beta, gamma, delta, epsilon, ann, max_iterations, test_set_path):
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma 
        self.delta = delta 
        self.epsilon = epsilon
        self.swarm = [generate_random_particle(id, 1, ann.neurons) for id in range(swarm_size)]
        #print(self.swarm)
        self.best = None
        self.ann = ann
        self.max_iterations = max_iterations
        self.test_set = pd.read_csv(test_set_path, sep='\t', header=None, names=['x', 'y'])
        self.n_informants = 6 #https://dl.acm.org/citation.cfm?doid=2330163.2330168

        for p in self.swarm:
            p.select_informants(self.swarm, self.n_informants)

    def execute(self):
        i = 0
        solution_found = False 
        while i < self.max_iterations and not solution_found:
            for particle in self.swarm:
                self.assess_fitness(particle)
                #print(particle.fitness)
                if self.best is None or particle.fitness < self.best.best_fitness:
                    self.best = particle
                if (particle.fitness < particle.best_fitness):
                    particle.best_fitness = particle.fitness
                    particle.best_fitness_position = particle.position

            x_swarm = self.get_fittest_position()
            for particle in self.swarm:
                new_speed = [np.zeros(particle.speed[i].shape) for i in range(len(particle.speed))]
                #new_speed = np.zeros(particle.speed.shape)
                x_fit = particle.best_fitness_position
                x_inf = particle.get_previous_fittest_of_informants()
                for l in range(len(particle.position)):
                    for j in range(len(particle.position[l])):
                        for k in range(len(particle.position[l][j])):
                            b = random.uniform(0, self.beta)
                            c = random.uniform(0, self.gamma)
                            d = random.uniform(0, self.delta)
                            new_speed[l][j][k] = self.alpha * particle.speed[l][j][k] + b * (x_fit[l][j][k] - particle.position[l][j][k]) + c * (x_inf[l][j][k] - particle.position[l][j][k]) + d * (x_swarm[l][j][k] - particle.position[l][j][k])
                particle.speed = new_speed

                for particle in self.swarm:
                    particle.update_position(self.epsilon)

            if self.best.fitness == 0:
                print("Solution found")
                solution_found = True

            print("Best fitness so far: {}".format(self.best.best_fitness))
            i+=1 
        
        self.plot_result()

    def assess_fitness(self, particle):
        #print('Testing particle {}'.format(particle.id))
        graph = []
        self.ann.set_weights(particle.position)
        mse = 0
        n = len(self.test_set)
        #print(self.test_set)
        for _, row in self.test_set.iterrows():
            #print(row[1])
            d = row[1]
            x_i = [row[0]]
            u = self.ann.process(x_i)
            graph.append(u)
            mse_i = (d - u) ** 2
            mse = mse + mse_i
        particle.fitness = mse / n
        particle.best_fitness_graph = graph
        #print(particle.fitness)

    def get_fittest_position(self):
        fittest_p = self.swarm[0]
        for p in self.swarm:
            if p.fitness < fittest_p.fitness:
                fittest_p = p
        return fittest_p.position

    def plot_result(self):
        x = self.test_set['x']
        y = self.test_set['y']
        g = self.best.best_fitness_graph

        plt.plot(x,y,x,g)
        plt.show()

