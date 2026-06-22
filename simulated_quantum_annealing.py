import numpy as np
import networkx as nx
from neal import SimulatedAnnealingSampler
from typing import Tuple, List
from collections import defaultdict
from config import SQA_PARAMS

class SQAOptimizer:
    def __init__(self, G: nx.DiGraph, source: int, target: int, capacity: float = None):
        self.G = G
        self.source = source
        self.target = target
        self.capacity = capacity
        self.edges = list(G.edges())
        self.edge_to_idx = {edge: i for i, edge in enumerate(self.edges)}
        
    def _calculate_dynamic_penalty(self) -> float:
        """Calculates a tighter structural constraint bound to prevent objective drowning."""
        edge_costs = [d['cost'] for u, v, d in self.G.edges(data=True)]
        max_positive = max((c for c in edge_costs if c > 0), default=1.0)
        min_negative = min((c for c in edge_costs if c < 0), default=0.0)
        
        # A valid simple path will cross at most len(G.nodes) steps
        max_path_cost = max_positive * len(self.G.nodes())
        max_path_savings = abs(min_negative) * len(self.G.nodes())
        
        # Tighter penalty threshold that protects the objective function precision
        return max_path_savings + max_path_cost + 25.0

    def build_qubo(self, mu: float = 0.0):
        """Constructs the QUBO matrix with structural lambda and a linear Lagrangian penalty."""
        Q = defaultdict(float)
        lambda_penalty = self._calculate_dynamic_penalty()
        
        # 1. Objective Function (Minimize Cost + mu * Resource)
        for edge, idx in self.edge_to_idx.items():
            u, v = edge
            cost = self.G[u][v]['cost']
            resource = self.G[u][v]['resource'] if self.capacity is not None else 0.0
            Q[(idx, idx)] += cost + mu * resource
            
        # 2. Structural Flow Conservation Constraints (Lambda)
        for node in self.G.nodes():
            if node == self.source:
                expected_flow = 1
            elif node == self.target:
                expected_flow = -1
            else:
                expected_flow = 0
                
            edges_out = [(node, v) for v in self.G.successors(node)]
            edges_in = [(u, node) for u in self.G.predecessors(node)]
            
            for edge in edges_out:
                idx = self.edge_to_idx[edge]
                Q[(idx, idx)] += lambda_penalty * (1 - 2 * expected_flow)
            for edge in edges_in:
                idx = self.edge_to_idx[edge]
                Q[(idx, idx)] += lambda_penalty * (1 + 2 * expected_flow)
                
            for i in range(len(edges_out)):
                for j in range(i + 1, len(edges_out)):
                    idx1, idx2 = sorted([self.edge_to_idx[edges_out[i]], self.edge_to_idx[edges_out[j]]])
                    Q[(idx1, idx2)] += 2 * lambda_penalty
            for i in range(len(edges_in)):
                for j in range(i + 1, len(edges_in)):
                    idx1, idx2 = sorted([self.edge_to_idx[edges_in[i]], self.edge_to_idx[edges_in[j]]])
                    Q[(idx1, idx2)] += 2 * lambda_penalty
            for e_out in edges_out:
                for e_in in edges_in:
                    idx1, idx2 = sorted([self.edge_to_idx[e_out], self.edge_to_idx[e_in]])
                    Q[(idx1, idx2)] -= 2 * lambda_penalty

        return Q

    def run(self) -> Tuple[float, List[int], bool]:
        sampler = SimulatedAnnealingSampler()
        if len(self.edges) > 1000:
            return float('inf'), [], False

        mu = 0.0  
        max_feedback_loops = 12 if self.capacity is not None else 1
        alpha = 0.4  
        
        last_valid_path = []
        last_valid_cost = float('inf')
        min_violation = float('inf')

        for loop_idx in range(max_feedback_loops):
            qubo = self.build_qubo(mu=mu)
            
            sampleset = sampler.sample_qubo(
                qubo, 
                num_reads=SQA_PARAMS["num_reads"], 
                num_sweeps=SQA_PARAMS["num_sweeps"],
                beta_range=SQA_PARAMS["beta_range"]
            )
            
            current_loop_violation = 0.0
            found_structural_path = False
            
            for datum in sampleset.data(['sample']):
                sample = datum.sample
                
                # Build an explicit adjacency map of active selections to handle branching
                active_successors = defaultdict(list)
                for edge, idx in self.edge_to_idx.items():
                    if sample[idx] == 1:
                        u, v = edge
                        active_successors[u].append(v)
                
                # Trace the structural path safely out of the sample noise
                node_path = [self.source]
                current = self.source
                valid_flow = True
                visited = {self.source}
                
                while current != self.target:
                    if current not in active_successors or not active_successors[current]:
                        valid_flow = False
                        break
                    
                    # Pick the primary path forward
                    next_node = active_successors[current][0]
                    
                    # Cycle breakout guard for noisy states
                    if next_node in visited:
                        valid_flow = False
                        break
                        
                    node_path.append(next_node)
                    visited.add(next_node)
                    current = next_node
                
                if valid_flow:
                    # Calculate clean metrics EXCLUSIVELY along the true path sequence
                    clean_cost = 0.0
                    clean_res = 0.0
                    for i in range(len(node_path) - 1):
                        u, v = node_path[i], node_path[i+1]
                        clean_cost += self.G[u][v]['cost']
                        clean_res += self.G[u][v]['resource']
                    
                    # Evaluate condition based on targeted measurements
                    if self.capacity is None or clean_res <= self.capacity:
                        return clean_cost, node_path, True
                    
                    if not found_structural_path:
                        found_structural_path = True
                        current_loop_violation = clean_res - self.capacity
                    
                    violation = clean_res - self.capacity
                    if violation < min_violation:
                        min_violation = violation
                        last_valid_path = node_path
                        last_valid_cost = clean_cost
            
            # Feedback step
            if found_structural_path and current_loop_violation > 0:
                mu += alpha * current_loop_violation
            else:
                mu += 2.5

        if last_valid_path:
            return last_valid_cost, last_valid_path, False
            
        return float('inf'), [], False