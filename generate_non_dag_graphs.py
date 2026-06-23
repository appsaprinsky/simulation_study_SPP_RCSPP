import json
import random
import networkx as nx
from pathlib import Path
from config import GRAPHS_DIR, GRAPH_SIZES_NON_DAG, MASTER_SEED

def generate_benchmark_graphs_non_dag():
    # Set seed for reproducibility
    random.seed(MASTER_SEED)
    
    # Define and create the new folder for Non-DAG instances
    GRAPHS_NON_DAG_DIR = GRAPHS_DIR.parent / "graphs_non_dag"
    GRAPHS_NON_DAG_DIR.mkdir(exist_ok=True, parents=True)
    
    for size in GRAPH_SIZES_NON_DAG:
        G = nx.DiGraph()
        G.add_nodes_from(range(size))
        
        # 1. Build a Strongly Connected Backbone (Forward + Backward Chains)
        # Forward chain guarantees a baseline path from Source (0) to Target (size-1)
        for i in range(size - 1):
            G.add_edge(i, i + 1)
            
        # Backward chain guarantees cycles and full strong connectivity ("really connected")
        for i in range(size - 1, 0, -1):
            G.add_edge(i, i - 1)
            
        # Add a giant feedback loop from target back to source
        G.add_edge(size - 1, 0)
        
        # 2. Inject Rugged & Random Cross-Edges (both forward skips and backward loops)
        # Higher density for smaller graphs, controlled density for larger ones
        edge_probability = 0.3 if size <= 20 else (1.5 / (size ** 0.5))
        
        for u in range(size):
            for v in range(size):
                if u != v and not G.has_edge(u, v):
                    if random.random() < edge_probability:
                        G.add_edge(u, v)

        # 3. Assign highly variable, deceptive costs and resource constraints
        for u, v in G.edges():
            # Allows negative costs to simulate deceptive/rewarding cycles
            G[u][v]['cost'] = random.uniform(-10.0, 25.0)
            G[u][v]['resource'] = random.uniform(1.0, 10.0)
            
        # 4. Verify connectivity constraints
        assert nx.is_strongly_connected(G), "Graph generation failed strong connectivity check!"
        assert not nx.is_directed_acyclic_graph(G), "Graph is mistakenly a DAG!"

        # 5. Calculate a challenging resource capacity bound
        # Using a proxy path length similar to the layered configuration
        estimated_path_length = max(3, size // 4)
        capacity = estimated_path_length * 5.5 * 0.8
        
        graph_data = {
            "num_nodes": size,
            "source": 0,
            "target": size - 1,
            "capacity": capacity,
            "edges": [
                {
                    "u": int(u), 
                    "v": int(v), 
                    "cost": float(d["cost"]), 
                    "resource": float(d["resource"])
                } 
                for u, v, d in G.edges(data=True)
            ]
        }
        
        file_path = GRAPHS_NON_DAG_DIR / f"graph_{size}.json"
        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)
            
        print(f"Generated rugged Non-DAG (Strongly Connected): {file_path.name} -> saved to {GRAPHS_NON_DAG_DIR.name}/")

if __name__ == "__main__":
    generate_benchmark_graphs_non_dag()