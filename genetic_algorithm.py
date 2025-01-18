import numpy as np
import math

class GeneticAlgorithm:
    def __init__(self, function, x_min, x_max, dx, population_size, generations, crossover_prob, mutation_prob):
        self.function = function
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.setup_parameters()

    def setup_parameters(self):
        self.n_points = int((self.x_max - self.x_min) / self.dx) + 1
        self.n_bits = math.ceil(math.log2(self.n_points))
        self.dx_system = (self.x_max - self.x_min) / (2**self.n_bits - 1)

    def initialize_population(self):
        return [''.join(np.random.choice(['0', '1']) for _ in range(self.n_bits)) 
                for _ in range(self.population_size)]

    def fitness(self, individual):
        decimal = int(individual, 2)
        x = self.x_min + decimal * self.dx_system
        return self.function(x)

    def select_best(self, population):
        fitness_values = [(individual, self.fitness(individual)) for individual in population]
        population_sorted = [ind for ind, _ in sorted(fitness_values, key=lambda x: x[1], reverse=True)]
        n_selected = max(2, len(population) // 2)
        return population_sorted[:n_selected]

    def crossover(self, population):
        fitness_values = [(individual, self.fitness(individual)) for individual in population]
        population_sorted = [ind for ind, _ in sorted(fitness_values, key=lambda x: x[1], reverse=True)]
        
        new_population = []
        pairs = []
        
        for i in range(len(population_sorted)):
            if np.random.random() <= self.crossover_prob:
                j = np.random.randint(0, i+1)
                pairs.append((i, j))
                
                parent1 = population_sorted[i]
                parent2 = population_sorted[j]
                
                crossover_point = np.random.randint(1, self.n_bits)
                
                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]
                
                new_population.extend([child1, child2])
            else:
                new_population.append(population_sorted[i])
        
        while len(new_population) > len(population):
            new_population.pop()
        
        while len(new_population) < len(population):
            idx = np.random.randint(0, len(new_population))
            new_population.append(new_population[idx])
        
        return new_population, len(pairs)

    def mutate(self, population):
        mutated_population = []
        total_mutations = 0
        total_mutated_bits = 0
        
        for individual in population:
            if np.random.random() <= self.mutation_prob:
                new_individual = list(individual)
                
                for i in range(self.n_bits):
                    if np.random.random() <= self.mutation_prob:
                        new_individual[i] = '1' if new_individual[i] == '0' else '0'
                        total_mutated_bits += 1
                
                new_individual = ''.join(new_individual)
                total_mutations += 1
                mutated_population.append(new_individual)
            else:
                mutated_population.append(individual)
        
        return mutated_population, total_mutations, total_mutated_bits

    def get_best_and_worst(self, population):
        fitness_values = [(individual, self.fitness(individual)) for individual in population]
        sorted_individuals = sorted(fitness_values, key=lambda x: x[1], reverse=True)
        return sorted_individuals[0][0], sorted_individuals[-1][0]

    def decode_solution(self, binary_string):
        decimal = int(binary_string, 2)
        x = self.x_min + decimal * self.dx_system
        fx = self.fitness(binary_string)
        return x, fx