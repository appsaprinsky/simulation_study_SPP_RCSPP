import random
import numpy as np
import networkx as nx
from typing import Tuple, List
from config import GA_PARAMS, PENALTY_MULTIPLIER

class GeneticAlgorithm:
    def __init__(self, G: nx.DiGraph, source: int, target: int, capacity: float = None):
        self.G = G
        self.source = source
        self.target = target
        self.capacity = capacity
        self.nodes = list(G.nodes())
        self.num_nodes = len(self.nodes)

    def decode(self, priority_vector: np.ndarray) -> Tuple[float, float, List[int]]:
        """Decodes priorities into a valid path. Returns (fitness, resource, path)."""
        current = self.source
        path = [current]
        total_cost = 0.0
        total_res = 0.0
        
        while current != self.target:
            neighbors = list(self.G.successors(current))
            unvisited = [n for n in neighbors if n not in path]
            
            if not unvisited:
                return float('inf'), float('inf'), path # Dead end
                
            # Select unvisited neighbor with highest priority
            next_node = max(unvisited, key=lambda n: priority_vector[self.nodes.index(n)])
            
            total_cost += self.G[current][next_node]['cost']
            total_res += self.G[current][next_node]['resource']
            current = next_node
            path.append(current)
            
        fitness = total_cost
        if self.capacity is not None and total_res > self.capacity:
            fitness += (total_res - self.capacity) * PENALTY_MULTIPLIER
            
        return fitness, total_res, path

    def run(self) -> Tuple[float, List[int], bool]:
        pop = np.random.rand(GA_PARAMS["pop_size"], self.num_nodes)
        best_cost = float('inf')
        best_path = []
        is_feasible = False
        
        for _ in range(GA_PARAMS["generations"]):
            fitnesses = []
            for ind in pop:
                fit, res, pth = self.decode(ind)
                fitnesses.append(fit)
                if fit < best_cost and (self.capacity is None or res <= self.capacity):
                    best_cost = fit
                    best_path = pth
                    is_feasible = True
                    
            # Selection (Tournament)
            indices = np.argsort(fitnesses)
            parents = pop[indices[:GA_PARAMS["pop_size"] // 2]]
            
            # Crossover & Mutation
            next_gen = list(parents)
            while len(next_gen) < GA_PARAMS["pop_size"]:
                p1, p2 = random.choices(parents, k=2)
                # Uniform crossover
                mask = np.random.rand(self.num_nodes) > GA_PARAMS["crossover_rate"]
                child = np.where(mask, p1, p2)
                
                # Mutation
                if random.random() < GA_PARAMS["mutation_rate"]:
                    mutate_idx = random.randint(0, self.num_nodes - 1)
                    child[mutate_idx] = random.random()
                    
                next_gen.append(child)
                
            pop = np.array(next_gen)
            
        return best_cost, best_path, is_feasible