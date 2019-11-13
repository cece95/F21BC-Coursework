import numpy.random as rand
import numpy as np
import pandas as pd  
import random
import math
import matplotlib.pyplot as plt
from Particle import Particle

#Generate random particle to init PSO algorithm
def generate_random_particle(_id, input_size, neurons):
    position = []
    speed = []
    n_neurons = sum(neurons)
    n_weights = input_size * neurons[0]
    for i in range(len(neurons) - 1):
        n_weights = n_weights + neurons[i]*neurons[i+1] 
    total_n_values = n_weights + n_neurons # give the PSO the possibility to select the activation functions 
    position = 2 * rand.random_sample(total_n_values)
    speed = np.zeros(total_n_values)
    return Particle(_id, position, speed, n_weights, n_neurons)


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
        self.test_set = pd.read_csv(test_set_path, sep='\s+|\t+|\s+\t+|\t+\s+', header=None, names=['x', 'y'])
        self.n_informants = 6 #https://dl.acm.org/citation.cfm?doid=2330163.2330168

        for p in self.swarm:
            p.select_informants(self.swarm, self.n_informants)

    def execute(self):
        i = 0
        solution_found = False 
        while i < self.max_iterations and not solution_found:
            #for particle in self.swarm:
                #print('{} | Fitness: {}'.format(particle.position, particle.fitness))
            for particle in self.swarm:
                self.assess_fitness(particle)
                if (particle.fitness < particle.best_fitness):
                    particle.best_fitness = particle.fitness
                    particle.best_fitness_position = particle.position

                if self.best is None or particle.fitness < self.best.best_fitness:
                    self.best = particle
                    self.best_fitness = particle.fitness

            x_swarm = self.get_fittest_position()
            for particle in self.swarm:
                new_speed = np.zeros(particle.speed.shape)
                #new_speed = np.zeros(particle.speed.shape)
                x_fit = particle.best_fitness_position
                x_inf = particle.get_previous_fittest_of_informants()
                for l in range(len(particle.position)):
                    b = random.uniform(0, self.beta)
                    c = random.uniform(0, self.gamma)
                    d = random.uniform(0, self.delta)
                    new_speed[l] = self.alpha * particle.speed[l] + b * (x_fit[l] - particle.position[l]) + c * (x_inf[l] - particle.position[l]) + d * (x_swarm[l] - particle.position[l])
                particle.speed = new_speed
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
        old_fitness = particle.best_fitness
        self.ann.set_values(particle.position)
        mse = 0
        n = len(self.test_set)
        #print(self.test_set.head(5))
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
        if (particle.fitness < old_fitness):
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

        print(self.best.best_fitness_position)

        plt.plot(x,y,x,g)
        plt.show()

