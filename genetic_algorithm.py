import numpy as np
import math

class GeneticAlgorithm:
    def __init__(self, function, x_min, x_max, dx, min_population, max_population, generations, crossover_prob, mutation_prob, bit_mutation_prob):
        self.function = function
        self.x_min = x_min
        self.x_max = x_max
        self.dx = dx
        self.min_population = min_population
        self.max_population = max_population
        self.population_size = max_population
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.bit_mutation_prob = bit_mutation_prob
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

    def get_population_stats(self, population):
        x_values = []
        fitness_values = []
        for individual in population:
            x, fx = self.decode_solution(individual)
            x_values.append(x)
            fitness_values.append(fx)
        return x_values, fitness_values

    def select_best(self, population):
        fitness_values = [(individual, self.fitness(individual)) for individual in population]
        population_sorted = [ind for ind, _ in sorted(fitness_values, key=lambda x: x[1], reverse=True)]
        n_selected = max(self.min_population, len(population) // 2)
        return population_sorted[:n_selected]
    
    def make_pairs(self, population):
        population_size = len(population)
        half_size = population_size // 2
        
        selected_indices = np.random.choice(population_size, half_size, replace=False)
        selected_population = [population[i] for i in selected_indices]
        
        fitness_values = [(individual, self.fitness(individual)) for individual in selected_population]
        selected_sorted = [ind for ind, _ in sorted(fitness_values, key=lambda x: x[1])]
        
        pairs = []
        for i in range(0, len(selected_sorted)-1, 2):
            if np.random.random() <= self.crossover_prob:
                pairs.append((selected_sorted[i], selected_sorted[i+1]))
        
        return pairs, [i for i in range(population_size) if i not in selected_indices]

    def crossover(self, population):
        pairs, remaining_indices = self.make_pairs(population)
        new_population = []
        
        for parent1, parent2 in pairs:
            child1 = list(parent1)
            child2 = list(parent2)
            
            for bit_pos in range(self.n_bits):
                if bit_pos % 2 == 1:  
                    child1[bit_pos], child2[bit_pos] = child2[bit_pos], child1[bit_pos]
            
            new_population.extend([''.join(child1), ''.join(child2)])
        
        new_population.extend([population[i] for i in remaining_indices])
        return new_population, len(pairs)

    def prune_population(self, population):
        if len(population) <= self.max_population:
            return population
            
        unique_population = list(dict.fromkeys(population))
        
        fitness_values = [(individual, self.fitness(individual)) for individual in unique_population]
        sorted_population = sorted(fitness_values, key=lambda x: x[1], reverse=True)
        best_individual = sorted_population[0][0]
        
        remaining_population = [ind for ind, _ in sorted_population[1:]]
        
        if len(remaining_population) + 1 > self.max_population:
            indices = np.random.choice(
                len(remaining_population), 
                size=self.max_population - 1, 
                replace=False
            )
            remaining_population = [remaining_population[i] for i in indices]
        
        final_population = [best_individual] + remaining_population
        
        return final_population

    def mutate(self, population):
        mutated_population = []
        total_mutations = 0
        total_mutated_bits = 0
        
        for individual in population:
            individual_genes = list(individual)
            individual_mutated = False
            
            if np.random.random() <= self.mutation_prob: 
                for i in range(self.n_bits):  
                    if np.random.random() <= self.bit_mutation_prob:  
                        swap_position = np.random.randint(0, self.n_bits)
                        individual_genes[i], individual_genes[swap_position] = individual_genes[swap_position], individual_genes[i]
                        total_mutated_bits += 1
                        individual_mutated = True
                
                if individual_mutated:
                    total_mutations += 1
            
            mutated_population.append(''.join(individual_genes))
        
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