import numpy.random as rand
import numpy as np
import pandas as pd  
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from Particle import Particle

#Initialization of the plots
fig = plt.figure(figsize=(20,10))
axes = [None, None, None]

def generate_random_particle(_id, input_size, neurons):
    """Function to generate random particle to init PSO algorithm"""
    position = []
    speed = []
    n_neurons = sum(neurons)
    n_weights = input_size * neurons[0]
    for i in range(len(neurons) - 1):
        n_weights = n_weights + neurons[i]*neurons[i+1] 
    total_n_values = n_weights + (2* n_neurons) - 1 # give the PSO the possibility to select the activation functions and bias, subtract one because the activation function is not needed for the last neuron 
    position = 2 * rand.random_sample(total_n_values) - 1
    speed = np.zeros(total_n_values)
    return Particle(_id, position, speed, n_weights, n_neurons)


class PSO:
    """Class that implements the PSO algorithm"""
    def __init__(self, swarm_size, n_informants, alpha_max, alpha_min, beta, gamma, delta, epsilon, ann, max_iterations, test_set_path, input_size):
        axes[1] = fig.add_subplot(132)
        axes[2] = fig.add_subplot(133)
        
        self.swarm_size = swarm_size
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.beta = beta 
        self.gamma = gamma 
        self.delta = delta 
        self.epsilon = epsilon
        self.swarm = [generate_random_particle(id, input_size, ann.neurons) for id in range(swarm_size)] # init swarm
        self.best = None
        self.best_fitness = 1000 #initialise the error to an high value
        self.ann = ann
        self.max_iterations = max_iterations
        self.input_size = input_size
        self.n_informants = n_informants
        # Setup the dataset structure to expect and the function plots based on the input size
        if input_size == 1:
            columns = ['x', 'y']
            axes[0] = fig.add_subplot(131)
        else:
            columns = ['x1', 'x2', 'y']
            axes[0] = fig.add_subplot(131, projection='3d')
        self.test_set = pd.read_csv(test_set_path, sep='\s+|\t+|\s+\t+|\t+\s+', header=None, names=columns, engine='python')
        
        #init arrays used to plot the results during the execution
        self.error = []
        self.steps = []
        self.best_record = []

        #assign informants to each particle
        for p in self.swarm:
            p.select_informants(self.swarm, self.n_informants)

    def execute(self):
        """ Function to run the PSO algorithm"""
        anim = FuncAnimation(fig, self.step, frames=self.max_iterations, repeat=False)
        plt.show()

    def step(self, i):
        """ Wrapper to execute one step of the PSO algorithm and plot the indermediate results"""
        self.pso_step(i+1)
        self.plot_result()

    def pso_step(self, i):
        """ Execution of a step of the PSO algorithm as explained in the lectures slides """
        for particle in self.swarm:
                self.assess_fitness(particle)
                if self.best is None or particle.fitness < self.best_fitness:
                    self.best = particle
                    self.best_fitness = particle.fitness
                    self.best_fitness_position = particle.best_fitness_position

        x_swarm = self.best_fitness_position
        for particle in self.swarm:
            new_speed = np.zeros(particle.speed.shape)
            x_fit = particle.best_fitness_position
            x_inf = particle.get_previous_fittest_of_informants()
            for l in range(len(particle.position)):
                a = (self.alpha_max - self.alpha_min) * ((self.max_iterations - i) / self.max_iterations) + self.alpha_min
                b = random.uniform(0, self.beta)
                c = random.uniform(0, self.gamma)
                d = random.uniform(0, self.delta)
                new_speed[l] = a * particle.speed[l] + b * (x_fit[l] - particle.position[l]) + c * (x_inf[l] - particle.position[l]) + d * (x_swarm[l] - particle.position[l])
            particle.speed = new_speed
            particle.update_position(self.epsilon)

        self.steps.append(i)
        self.error.append(self.best_fitness)
        self.best_record.append(self.best.id)
        print("{} | Best fitness so far: {}".format(i, self.best_fitness))

    def assess_fitness(self, particle):
        """ Function to assess the fitness of a particle using MSE"""
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
            particle.best_fitness = particle.fitness
            particle.best_fitness_position = particle.position

    def plot_result(self):
        "Function to plot the intermediate results of the PSO algorithm"
        #clear the figure from previous step's results
        axes[0].clear()
        axes[1].clear()
        axes[2].clear()
        
        #Reconstruct the cleared plots
        axes[0].title.set_text('Functions')
        axes[1].title.set_text('MSE')
        axes[1].set_xlabel('Number of iterations')
        axes[1].set_ylabel('Mean Squared Error')
        axes[2].title.set_text('Best Particle')
        axes[2].set_xlabel('Number of iterations')
        axes[2].set_ylabel('Best Particle ID')


        #plot the results in a different manner depending on the input size
        if self.input_size == 1:
            x = self.test_set['x']
            y = self.test_set['y']
            g = self.best.best_fitness_graph

            axes[0].plot(x,g, label='Approximated Function')
            axes[0].plot(x,y, label='Desidered Function')
            axes[0].legend()
        else:
            x1 = self.test_set['x1']
            x2 = self.test_set['x2']
            y = self.test_set['y']
            g = self.best.best_fitness_graph

            axes[0].scatter(x1, x2, y, label='Desidered Function')
            axes[0].scatter(x1, x2, g, label='Approximated Function')
            axes[0].legend()

        #plot error
        axes[1].set_ylim([0, 0.1])
        axes[1].plot(self.steps, self.error)

        #plot the fittest particle
        axes[2].plot(self.steps, self.best_record)
        axes[2].set_ylim([0, self.swarm_size])
        