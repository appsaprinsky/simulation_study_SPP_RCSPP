import numpy as np
import networkx as nx
from typing import Tuple, List
from config import PSO_PARAMS, PENALTY_MULTIPLIER
from genetic_algorithm import GeneticAlgorithm  # Reuse decoder

class ParticleSwarmOptimization:
    def __init__(self, G: nx.DiGraph, source: int, target: int, capacity: float = None):
        self.decoder = GeneticAlgorithm(G, source, target, capacity)
        self.num_nodes = len(G.nodes())

    def run(self) -> Tuple[float, List[int], bool]:
        swarm_size = PSO_PARAMS["swarm_size"]
        positions = np.random.rand(swarm_size, self.num_nodes)
        velocities = np.random.uniform(-1, 1, (swarm_size, self.num_nodes))
        
        pbest_pos = np.copy(positions)
        pbest_fit = np.full(swarm_size, float('inf'))
        gbest_pos = np.zeros(self.num_nodes)
        gbest_fit = float('inf')
        best_path = []
        is_feasible = False
        
        for _ in range(PSO_PARAMS["iterations"]):
            for i in range(swarm_size):
                fit, res, pth = self.decoder.decode(positions[i])
                
                if fit < pbest_fit[i]:
                    pbest_fit[i] = fit
                    pbest_pos[i] = positions[i]
                    
                valid = self.decoder.capacity is None or res <= self.decoder.capacity
                if fit < gbest_fit and valid:
                    gbest_fit = fit
                    gbest_pos = positions[i]
                    best_path = pth
                    is_feasible = True
                    
            r1, r2 = np.random.rand(2)
            velocities = (PSO_PARAMS["w"] * velocities + 
                          PSO_PARAMS["c1"] * r1 * (pbest_pos - positions) + 
                          PSO_PARAMS["c2"] * r2 * (gbest_pos - positions))
                          
            positions = positions + velocities
            # Keep priorities in bounded range
            positions = np.clip(positions, 0.0, 1.0)
            
        return gbest_fit, best_path, is_feasible