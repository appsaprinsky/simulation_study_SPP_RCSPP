import json
import random
import networkx as nx
from pathlib import Path
from config import GRAPHS_DIR, GRAPH_SIZES, MASTER_SEED

def generate_benchmark_graphs():
    random.seed(MASTER_SEED)
    
    for size in GRAPH_SIZES:
        # Adjust sparsity based on scale
        p = 0.5 if size <= 20 else 0.15
        G = nx.gnp_random_graph(size, p, directed=True)
        
        # Guarantee a Hamiltonian-like path to ensure feasibility
        for i in range(size - 1):
            G.add_edge(i, i + 1)
            
        # Assign edge attributes
        for u, v in G.edges():
            G[u][v]['cost'] = random.uniform(-5.0, 15.0)  # Costs can be negative
            G[u][v]['resource'] = random.uniform(1.0, 10.0) # Resources strictly positive
            
        # Define Resource Budget (Capacity)
        # Strategy: Ensure capacity allows a direct path, but restricts dense random walks
        shortest_path_nodes = size
        avg_resource_per_edge = 5.5
        capacity = shortest_path_nodes * avg_resource_per_edge * 0.8
        
        # Format for JSON serialization
        graph_data = {
            "num_nodes": size,
            "source": 0,
            "target": size - 1,
            "capacity": capacity,
            "edges": [{"u": int(u), "v": int(v), "cost": float(d["cost"]), "resource": float(d["resource"])} 
                      for u, v, d in G.edges(data=True)]
        }
        
        file_path = GRAPHS_DIR / f"graph_{size}.json"
        with open(file_path, "w") as f:
            json.dump(graph_data, f, indent=4)
            
        print(f"Generated and saved: {file_path.name}")

if __name__ == "__main__":
    generate_benchmark_graphs()