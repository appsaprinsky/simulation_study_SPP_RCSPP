import json
import random
import networkx as nx
from pathlib import Path
from config import GRAPHS_DIR, GRAPH_SIZES, MASTER_SEED

def generate_benchmark_graphs():
    random.seed(MASTER_SEED)
    GRAPHS_DIR.mkdir(exist_ok=True, parents=True)
    
    for size in GRAPH_SIZES:
        G = nx.DiGraph()
        G.add_nodes_from(range(size))
        
        # 1. Create a Layered Structure to force combinatorial pathfinding
        num_layers = max(3, size // 4)
        layers = [[] for _ in range(num_layers)]
        
        for i in range(size):
            if i == 0:
                layers[0].append(i)
            elif i == size - 1:
                layers[-1].append(i)
            else:
                layer_idx = random.randint(1, num_layers - 2)
                layers[layer_idx].append(i)
                
        # Prevent empty layers to maintain connectivity
        for i in range(1, num_layers - 1):
            if not layers[i]:
                largest = max(range(1, num_layers - 1), key=lambda x: len(layers[x]))
                if len(layers[largest]) > 1:
                    layers[i].append(layers[largest].pop())
                    
        # 2. Connect layers (Rugged Topology)
        for i in range(num_layers - 1):
            for u in layers[i]:
                # Connect to 1-3 random nodes in the direct next layer
                targets = random.sample(layers[i+1], min(len(layers[i+1]), random.randint(1, 3)))
                for v in targets:
                    G.add_edge(u, v)
                    
                # Add deceptive "skip" edges (15% chance)
                if i < num_layers - 2 and random.random() < 0.15:
                    v_skip = random.choice(layers[i+2])
                    G.add_edge(u, v_skip)
        
        # 3. Assign highly variable, deceptive costs
        for u, v in G.edges():
            G[u][v]['cost'] = random.uniform(-10.0, 25.0)
            G[u][v]['resource'] = random.uniform(1.0, 10.0)
            
        # 4. Fallback: Guarantee at least one valid path exists
        if not nx.has_path(G, 0, size - 1):
            current = 0
            for i in range(1, num_layers):
                next_node = random.choice(layers[i])
                if not G.has_edge(current, next_node):
                    G.add_edge(current, next_node, cost=random.uniform(-5.0, 5.0), resource=random.uniform(1.0, 5.0))
                current = next_node

        # Calculate a restrictive but feasible resource capacity
        capacity = num_layers * 5.5 * 0.8
        
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
            
        print(f"Generated rugged Layered DAG: {file_path.name}")

if __name__ == "__main__":
    generate_benchmark_graphs()