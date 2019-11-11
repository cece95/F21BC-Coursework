import random
import math

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
        print(type(self.speed))
        new_position = self.position + self.speed * epsilon
        for i in range(len(new_position)):
            tmp_pos = new_position[i]
            while tmp_pos < -1 or tmp_pos > 1:    
                if tmp_pos > 1:
                    tmp_pos = 2 - tmp_pos
                elif tmp_pos < -1:
                    tmp_pos = -2 - tmp_pos
            new_position[i] = tmp_pos 