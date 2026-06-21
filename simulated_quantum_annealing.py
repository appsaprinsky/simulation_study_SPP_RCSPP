import numpy as np
import networkx as nx
from neal import SimulatedAnnealingSampler
from typing import Tuple, List
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
        """Calculates structural lambda penalty to prevent broken paths."""
        sum_negative_edges = sum(d['cost'] for u, v, d in self.G.edges(data=True) if d['cost'] < 0)
        max_positive_edge = max((d['cost'] for u, v, d in self.G.edges(data=True)), default=0)
        base_penalty = abs(sum_negative_edges) + max_positive_edge + 50.0
        scale_factor = max(1.0, len(self.G.nodes()) / 10.0)
        return base_penalty * scale_factor

    def build_qubo(self, mu: float = 0.0):
        """Constructs the QUBO matrix with structural lambda and a linear Lagrangian resource penalty."""
        from collections import defaultdict
        Q = defaultdict(float)
        
        lambda_penalty = self._calculate_dynamic_penalty()
        
        # 1. Objective Function (Minimize Cost + mu * Resource)
        # Linear Lagrangian modification preserves perfect matrix sparsity
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

        # Parameters for adaptive multiplier scaling (Lagrangian Relaxation Loop)
        mu = 0.0  
        max_feedback_loops = 10 if self.capacity is not None else 1
        alpha = 0.5  # Dynamic learning rate step size
        
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
            
            # Post-selection filtering across returned states sorted by energy
            for datum in sampleset.data(['sample']):
                sample = datum.sample
                cost, res = 0.0, 0.0
                path_edges = []
                
                for edge, idx in self.edge_to_idx.items():
                    if sample[idx] == 1:
                        u, v = edge
                        cost += self.G[u][v]['cost']
                        res += self.G[u][v]['resource']
                        path_edges.append(edge)
                
                # Trace structural path flow validity
                edge_dict = {u: v for u, v in path_edges}
                node_path = [self.source]
                current = self.source
                valid_flow = True
                
                while current != self.target:
                    if current not in edge_dict:
                        valid_flow = False
                        break
                    next_node = edge_dict[current]
                    node_path.append(next_node)
                    current = next_node
                
                if valid_flow:
                    if self.capacity is None or res <= self.capacity:
                        # SUCCESS: Path is structurally sound and respects the resource budget
                        return cost, node_path, True
                    
                    # Track metrics for the lowest-energy structural path of this specific iteration
                    if not found_structural_path:
                        found_structural_path = True
                        current_loop_violation = res - self.capacity
                    
                    # Track global fallback for the closest near-feasible path across all loops
                    violation = res - self.capacity
                    if violation < min_violation:
                        min_violation = violation
                        last_valid_path = node_path
                        last_valid_cost = cost
            
            # Subgradient adaptation step
            if found_structural_path and current_loop_violation > 0:
                mu += alpha * current_loop_violation
            else:
                # If the landscape didn't yield clean paths, steadily step up the pressure
                mu += 2.0

        # Fallback Strategy: If max iterations expire without a perfect match,
        # return the structural path that had the lowest overall capacity violation
        if last_valid_path:
            return last_valid_cost, last_valid_path, False
            
        return float('inf'), [], False