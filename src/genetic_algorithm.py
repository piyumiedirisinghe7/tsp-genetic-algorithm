import random
import numpy as np

class TSPGeneticAlgorithm:
    def __init__(self, distance_matrix, population_size=200, generations=1000, 
                 mutation_rate=0.015, tournament_size=5, elite_size=2):
        self.distance_matrix = distance_matrix
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elite_size = elite_size
        
    def create_individual(self):
        route = list(range(self.num_cities))
        random.shuffle(route)
        return route
    
    def create_population(self):
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_distance(self, route):
        total = 0
        for i in range(len(route)):
            from_city = route[i]
            to_city = route[(i+1) % len(route)]
            total += self.distance_matrix[from_city][to_city]
        return total
    
    def tournament_selection(self, population, fitness):
        selected = []
        for _ in range(len(population)):
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = [fitness[i] for i in tournament_indices]
            winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
            selected.append(population[winner_idx])
        return selected
    
    def roulette_selection(self, population, fitness):
        total_fitness = sum(1/f for f in fitness)
        probabilities = [(1/f) / total_fitness for f in fitness]
        selected = random.choices(population, weights=probabilities, k=len(population))
        return selected
    
    def rank_selection(self, population, fitness):
        sorted_indices = np.argsort(fitness)
        ranks = range(len(population), 0, -1)
        total_rank = sum(ranks)
        probabilities = [r/total_rank for r in ranks]
        selected_indices = random.choices(sorted_indices, weights=probabilities, k=len(population))
        return [population[i] for i in selected_indices]
    
    def ordered_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end+1] = parent1[start:end+1]
        
        position = 0
        for city in parent2:
            if city not in child:
                while child[position] != -1:
                    position += 1
                child[position] = city
        return child
    
    def pmx_crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        
        child = [-1] * size
        child[start:end+1] = parent1[start:end+1]
        
        mapping = {}
        for i in range(start, end+1):
            mapping[parent2[i]] = parent1[i]
        
        for i in range(size):
            if i < start or i > end:
                city = parent2[i]
                while city in mapping:
                    city = mapping[city]
                child[i] = city
        return child
    
    def cycle_crossover(self, parent1, parent2):
        size = len(parent1)
        child = [-1] * size
        visited = [False] * size
        
        cycle = 0
        while not all(visited):
            if cycle % 2 == 0:
                idx = visited.index(False)
                while not visited[idx]:
                    child[idx] = parent1[idx]
                    visited[idx] = True
                    idx = parent2.index(parent1[idx])
            else:
                idx = visited.index(False)
                while not visited[idx]:
                    child[idx] = parent2[idx]
                    visited[idx] = True
                    idx = parent2.index(parent1[idx])
            cycle += 1
        return child
    
    def mutate(self, route):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(route)), 2)
            route[i], route[j] = route[j], route[i]
        return route
    
    def mutate_inverse(self, route):
        if random.random() < self.mutation_rate:
            i, j = sorted(random.sample(range(len(route)), 2))
            route[i:j+1] = reversed(route[i:j+1])
        return route
    
    def two_opt(self, route):
        improved = True
        best = route.copy()
        best_distance = self.calculate_distance(best)
        
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1:
                        continue
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_distance = self.calculate_distance(new_route)
                    
                    if new_distance < best_distance:
                        best = new_route
                        best_distance = new_distance
                        improved = True
        return best
    
    def three_opt(self, route):
        best = route.copy()
        best_distance = self.calculate_distance(best)
        improved = True
        
        while improved:
            improved = False
            for i in range(1, len(route) - 3):
                for j in range(i + 2, len(route) - 1):
                    for k in range(j + 2, len(route)):
                        segments = [
                            best[:i], best[i:j], best[j:k], best[k:]
                        ]
                        
                        new_routes = [
                            segments[0] + segments[1] + segments[2] + segments[3],
                            segments[0] + segments[2] + segments[1] + segments[3],
                            segments[0] + segments[1][::-1] + segments[2] + segments[3],
                            segments[0] + segments[2][::-1] + segments[1] + segments[3],
                            segments[0] + segments[1] + segments[2][::-1] + segments[3],
                            segments[0] + segments[2] + segments[1][::-1] + segments[3],
                            segments[0] + segments[1][::-1] + segments[2][::-1] + segments[3],
                        ]
                        
                        for new_route in new_routes:
                            new_distance = self.calculate_distance(new_route)
                            if new_distance < best_distance:
                                best = new_route
                                best_distance = new_distance
                                improved = True
        return best
    
    def evolve(self, selection_type="tournament", crossover_type="ox", 
               mutation_type="swap", use_2opt=True, use_3opt=False):
        
        population = self.create_population()
        best_route = None
        best_distance = float('inf')
        history = []
        best_history = []
        
        for generation in range(self.generations):
            fitness = [self.calculate_distance(route) for route in population]
            
            gen_best_idx = np.argmin(fitness)
            gen_best_distance = fitness[gen_best_idx]
            
            if gen_best_distance < best_distance:
                best_distance = gen_best_distance
                best_route = population[gen_best_idx].copy()
            
            history.append(best_distance)
            best_history.append(gen_best_distance)
            
            if selection_type == "tournament":
                selected = self.tournament_selection(population, fitness)
            elif selection_type == "roulette":
                selected = self.roulette_selection(population, fitness)
            elif selection_type == "rank":
                selected = self.rank_selection(population, fitness)
            else:
                selected = population
            
            next_population = []
            
            elite_indices = np.argsort(fitness)[:self.elite_size]
            for idx in elite_indices:
                next_population.append(population[idx].copy())
            
            while len(next_population) < self.population_size:
                parent1, parent2 = random.sample(selected, 2)
                
                if crossover_type == "ox":
                    child1 = self.ordered_crossover(parent1, parent2)
                    child2 = self.ordered_crossover(parent2, parent1)
                elif crossover_type == "pmx":
                    child1 = self.pmx_crossover(parent1, parent2)
                    child2 = self.pmx_crossover(parent2, parent1)
                elif crossover_type == "cycle":
                    child1 = self.cycle_crossover(parent1, parent2)
                    child2 = self.cycle_crossover(parent2, parent1)
                else:
                    child1 = self.ordered_crossover(parent1, parent2)
                    child2 = self.ordered_crossover(parent2, parent1)
                
                if mutation_type == "swap":
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                elif mutation_type == "inverse":
                    child1 = self.mutate_inverse(child1)
                    child2 = self.mutate_inverse(child2)
                else:
                    child1 = self.mutate(child1)
                    child2 = self.mutate(child2)
                
                if use_2opt:
                    child1 = self.two_opt(child1)
                    child2 = self.two_opt(child2)
                
                if use_3opt:
                    child1 = self.three_opt(child1)
                    child2 = self.three_opt(child2)
                
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            population = next_population
            
            if generation % 100 == 0:
                print(f"Generation {generation}: Best = {best_distance:.2f}, Current Best = {gen_best_distance:.2f}")
        
        if use_2opt:
            best_route = self.two_opt(best_route)
            best_distance = self.calculate_distance(best_route)
        
        if use_3opt:
            best_route = self.three_opt(best_route)
            best_distance = self.calculate_distance(best_route)
        
        return best_route, best_distance, history, best_history