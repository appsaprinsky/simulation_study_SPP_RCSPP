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
        
    def build_qubo(self) -> dict:
        num_edges = len(self.edges)
        
        # Slack variables for capacity
        num_slacks = 0
        if self.capacity is not None:
            num_slacks = int(np.log2(max(1, int(self.capacity)))) + 1
            
        total_vars = num_edges + num_slacks
        Q = np.zeros((total_vars, total_vars))
        
        P_flow = SQA_PARAMS["flow_penalty"]
        P_cap = SQA_PARAMS["cap_penalty"]
        
        # 1. Objective Costs
        for edge, idx in self.edge_to_idx.items():
            u, v = edge
            Q[idx, idx] += self.G[u][v]['cost']
            
        # 2. Flow Conservation
        for node in self.G.nodes():
            b = 1 if node == self.source else (-1 if node == self.target else 0)
            in_e = [self.edge_to_idx[e] for e in self.G.in_edges(node)]
            out_e = [self.edge_to_idx[e] for e in self.G.out_edges(node)]
            
            all_e = in_e + out_e
            signs = {idx: 1 for idx in in_e}
            for idx in out_e: signs[idx] = -1
                
            for idx in in_e: Q[idx, idx] -= 2 * b * P_flow
            for idx in out_e: Q[idx, idx] += 2 * b * P_flow
                
            for i in range(len(all_e)):
                for j in range(i, len(all_e)):
                    idx_i, idx_j = all_e[i], all_e[j]
                    val = P_flow * signs[idx_i] * signs[idx_j]
                    if i == j: Q[idx_i, idx_i] += val
                    else: Q[min(idx_i, idx_j), max(idx_i, idx_j)] += 2 * val
                        
        # 3. Capacity Constraint
        if self.capacity is not None:
            slack_indices = [num_edges + k for k in range(num_slacks)]
            slack_weights = [2**k for k in range(num_slacks)]
            
            cap_vars = list(self.edge_to_idx.values()) + slack_indices
            cap_weights = [self.G[e[0]][e[1]]['resource'] for e in self.edges] + slack_weights
            
            for i, idx_i in enumerate(cap_vars):
                w_i = cap_weights[i]
                Q[idx_i, idx_i] -= 2 * self.capacity * w_i * P_cap
                for j in range(i, len(cap_vars)):
                    idx_j = cap_vars[j]
                    w_j = cap_weights[j]
                    val = P_cap * w_i * w_j
                    if i == j: Q[idx_i, idx_i] += val
                    else: Q[min(idx_i, idx_j), max(idx_i, idx_j)] += 2 * val

        return {(i, j): Q[i, j] for i in range(total_vars) for j in range(i, total_vars) if Q[i, j] != 0}

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
        
        # Reconstruct path and verify
        cost, res = 0.0, 0.0
        path_edges = []
        for edge, idx in self.edge_to_idx.items():
            if best_sample[idx] == 1:
                u, v = edge
                cost += self.G[u][v]['cost']
                res += self.G[u][v]['resource']
                path_edges.append(edge)
                
        # Basic structural verification
        is_feasible = True
        if self.capacity and res > self.capacity:
            is_feasible = False
            
        return cost, path_edges, is_feasible