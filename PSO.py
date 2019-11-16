import numpy.random as rand
import numpy as np
import pandas as pd  
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
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
    def __init__(self, swarm_size, alpha, beta, gamma, delta, epsilon, ann, max_iterations, test_set_path, input_size):
        self.swarm_size = swarm_size
        self.alpha = alpha
        self.beta = beta 
        self.gamma = gamma 
        self.delta = delta 
        self.epsilon = epsilon
        self.swarm = [generate_random_particle(id, input_size, ann.neurons) for id in range(swarm_size)]
        self.best = None
        self.ann = ann
        self.max_iterations = max_iterations
        self.input_size = input_size
        if input_size == 1:
            columns = ['x', 'y']
        else:
            columns = ['x1', 'x2', 'y']
        self.test_set = pd.read_csv(test_set_path, sep='\s+|\t+|\s+\t+|\t+\s+', header=None, names=columns)
        self.n_informants = 6 #https://dl.acm.org/citation.cfm?doid=2330163.2330168

        for p in self.swarm:
            p.select_informants(self.swarm, self.n_informants)

    def execute(self):
        fig, axes = plt.subplots()
        anim = FuncAnimation(fig, update_plot, frames=self.max_iterations)

    def update_plot(self):
        self.pso_step()
        self.plot_result()


    def pso_step(self):
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
                x_fit = particle.best_fitness_position
                x_inf = particle.get_previous_fittest_of_informants()
                for l in range(len(particle.position)):
                    b = random.uniform(0, self.beta)
                    c = random.uniform(0, self.gamma)
                    d = random.uniform(0, self.delta)
                    new_speed[l] = self.alpha * particle.speed[l] + b * (x_fit[l] - particle.position[l]) + c * (x_inf[l] - particle.position[l]) + d * (x_swarm[l] - particle.position[l])
                particle.speed = new_speed
                particle.update_position(self.epsilon)

            print("Best fitness so far: {}".format(self.best.best_fitness))

    def assess_fitness(self, particle):
        graph = []
        old_fitness = particle.best_fitness
        self.ann.set_values(particle.position)
        mse = 0
        n = len(self.test_set)
        for _, row in self.test_set.iterrows():
            if self.input_size == 1:
                x_i = [row[0]]
                d = row[1]
            else:
                x_i = [row[0], row[1]]
                d = row[2]

            u = self.ann.process(x_i)
            graph.append(u)
            mse_i = (d - u) ** 2
            mse = mse + mse_i
        particle.fitness = mse / n
        if (particle.fitness < old_fitness):
            particle.best_fitness_graph = graph

    def get_fittest_position(self):
        fittest_p = self.swarm[0]
        for p in self.swarm:
            if p.fitness < fittest_p.fitness:
                fittest_p = p
        return fittest_p.position

    def plot_result(self):
        if self.input_size == 1:
            x = self.test_set['x']
            y = self.test_set['y']
            g = self.best.best_fitness_graph

            axes.plot(x,y,x,g)
        else:
            x1 = self.test_set['x1']
            x2 = self.test_set['x2']
            y = self.test_set['y']
            g = self.best.best_fitness_graph

            ax = fig.gca(projection='3d')
            ax.scatter(x1, x2, y)
            ax.scatter(x1, x2, g)