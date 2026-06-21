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
        """
        Calculates a strict constraint penalty (lambda) that outweighs any 
        possible benefit the quantum sampler could get from illegally picking negative edges.
        """
        # Calculate the absolute max reward for cheating
        sum_negative_edges = sum(d['cost'] for u, v, d in self.G.edges(data=True) if d['cost'] < 0)
        max_positive_edge = max((d['cost'] for u, v, d in self.G.edges(data=True)), default=0)
        
        # Penalty = |Max Cheat Reward| + Highest Edge + Buffer
        base_penalty = abs(sum_negative_edges) + max_positive_edge + 50.0
        
        # Scale for larger graphs to maintain quantum pressure
        scale_factor = max(1.0, len(self.G.nodes()) / 10.0)
        return base_penalty * scale_factor

    def build_qubo(self):
        """Constructs the QUBO matrix using a dynamically scaled lambda penalty."""
        from collections import defaultdict
        Q = defaultdict(float)
        
        # 1. Get the dynamic lambda penalty
        lambda_penalty = self._calculate_dynamic_penalty()
        
        # 2. Apply Objective Function (Minimize Cost)
        for edge, idx in self.edge_to_idx.items():
            u, v = edge
            Q[(idx, idx)] += self.G[u][v]['cost']
            
        # 3. Apply Flow Conservation Constraints
        for node in self.G.nodes():
            if node == self.source:
                expected_flow = 1
            elif node == self.target:
                expected_flow = -1
            else:
                expected_flow = 0
                
            edges_out = [(node, v) for v in self.G.successors(node)]
            edges_in = [(u, node) for u in self.G.predecessors(node)]
            
            # Linear terms
            for edge in edges_out:
                idx = self.edge_to_idx[edge]
                Q[(idx, idx)] += lambda_penalty * (1 - 2 * expected_flow)
                
            for edge in edges_in:
                idx = self.edge_to_idx[edge]
                Q[(idx, idx)] += lambda_penalty * (1 + 2 * expected_flow)
                
            # Quadratic terms (out * out)
            for i in range(len(edges_out)):
                for j in range(i + 1, len(edges_out)):
                    idx1, idx2 = sorted([self.edge_to_idx[edges_out[i]], self.edge_to_idx[edges_out[j]]])
                    Q[(idx1, idx2)] += 2 * lambda_penalty
                    
            # Quadratic terms (in * in)
            for i in range(len(edges_in)):
                for j in range(i + 1, len(edges_in)):
                    idx1, idx2 = sorted([self.edge_to_idx[edges_in[i]], self.edge_to_idx[edges_in[j]]])
                    Q[(idx1, idx2)] += 2 * lambda_penalty
                    
            # Cross terms (out * in) - These subtract penalty because 1 in, 1 out = balanced flow
            for e_out in edges_out:
                for e_in in edges_in:
                    idx1, idx2 = sorted([self.edge_to_idx[e_out], self.edge_to_idx[e_in]])
                    Q[(idx1, idx2)] -= 2 * lambda_penalty

        return Q

    def run(self) -> Tuple[float, List[int], bool]:
        qubo = self.build_qubo()
        sampler = SimulatedAnnealingSampler()
        
        # For large graphs (>50 nodes), SQA QUBO matrix becomes massive. Added safety fallback.
        if len(self.edges) > 1000:
            return float('inf'), [], False

        sampleset = sampler.sample_qubo(
            qubo, 
            num_reads=SQA_PARAMS["num_reads"], 
            num_sweeps=SQA_PARAMS["num_sweeps"],
            beta_range=SQA_PARAMS["beta_range"]
        )
        
        best_sample = sampleset.first.sample
        
        cost, res = 0.0, 0.0
        path_edges = []
        for edge, idx in self.edge_to_idx.items():
            if best_sample[idx] == 1:
                u, v = edge
                cost += self.G[u][v]['cost']
                res += self.G[u][v]['resource']
                path_edges.append(edge)
                
        # --- NEW: Convert selected edges into a sequential list of nodes ---
        edge_dict = {u: v for u, v in path_edges}
        node_path = [self.source]
        current = self.source
        is_feasible = True

        # Trace the path from source to target to ensure structural validity
        while current != self.target:
            if current not in edge_dict:
                # Flow conservation was violated by the quantum sampler (broken path)
                is_feasible = False
                break
            
            next_node = edge_dict[current]
            node_path.append(next_node)
            current = next_node
                
        # Verify capacity constraint
        if self.capacity and res > self.capacity:
            is_feasible = False
            
        return cost, node_path, is_feasible