import networkx as nx
from typing import Tuple, Optional, List

class ExactSolvers:
    @staticmethod
    def solve_spp(G: nx.DiGraph, source: int, target: int) -> Tuple[float, List[int]]:
        """Solves Standard SPP using Bellman-Ford (allows negative weights)."""
        try:
            path = nx.bellman_ford_path(G, source, target, weight="cost")
            cost = nx.bellman_ford_path_length(G, source, target, weight="cost")
            return cost, path
        except nx.NetworkXUnbounded:
            return float('-inf'), []  # Negative cycle detected
        except nx.NetworkXNoPath:
            return float('inf'), []

    @staticmethod
    def solve_rcspp(G: nx.DiGraph, source: int, target: int, capacity: float) -> Tuple[float, List[int]]:
        """Solves RCSPP using an exact Labeling Algorithm with dominance pruning."""
        # labels[node] = [(cost, resource, path_history)]
        labels = {n: [] for n in G.nodes()}
        labels[source].append((0.0, 0.0, [source]))
        
        queue = [source]
        
        while queue:
            u = queue.pop(0)
            
            for v in G.successors(u):
                edge_cost = G[u][v]['cost']
                edge_res = G[u][v]['resource']
                
                updated = False
                for c_u, r_u, path_u in labels[u]:
                    c_new = c_u + edge_cost
                    r_new = r_u + edge_res
                    
                    if r_new <= capacity and v not in path_u:
                        # Dominance check
                        dominated = False
                        for c_v, r_v, _ in labels[v]:
                            if c_v <= c_new and r_v <= r_new:
                                dominated = True
                                break
                        
                        if not dominated:
                            # Prune worse labels
                            labels[v] = [(c, r, p) for c, r, p in labels[v] if not (c_new <= c and r_new <= r)]
                            labels[v].append((c_new, r_new, path_u + [v]))
                            updated = True
                            
                if updated and v not in queue:
                    queue.append(v)
                    
        if not labels[target]:
            return float('inf'), []
            
        best_label = min(labels[target], key=lambda x: x[0])
        return best_label[0], best_label[2]