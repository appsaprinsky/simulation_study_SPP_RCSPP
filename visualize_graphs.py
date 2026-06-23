import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from config import GRAPHS_DIR, FIGURES_DIR

def get_optimal_layout(G: nx.DiGraph):
    """Determines the best visual layout based on the graph's structural topology."""
    if nx.is_directed_acyclic_graph(G):
        layers = list(nx.topological_generations(G))
        for layer, nodes in enumerate(layers):
            for node in nodes:
                G.nodes[node]["layer"] = layer
        return nx.multipartite_layout(G, subset_key="layer")
    else:
        return nx.spring_layout(G, k=2.0/((G.number_of_nodes())**0.5), seed=42)

def plot_solution(G: nx.DiGraph, path: list, title: str, out_path: Path, capacity: float = None):
    """Overlays the selected solution path onto the graph topology."""
    plt.figure(figsize=(10, 8))
    
    pos = get_optimal_layout(G)
    
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, edgecolors='black')
    
    if path:
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightgreen', node_size=400, edgecolors='black')
        
    nx.draw_networkx_edges(
        G, pos, arrowstyle="->", arrowsize=10, alpha=0.3, 
        edge_color="gray", connectionstyle="arc3,rad=0.1"
    )
    
    path_edges = list(zip(path[:-1], path[1:])) if path else []
    if path_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=path_edges, edge_color='red', 
            width=2.5, arrowstyle="->", arrowsize=15, connectionstyle="arc3,rad=0.1"
        )
        
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    if len(G.nodes()) <= 20:
        edge_labels = {(u, v): f"C:{d['cost']:.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, label_pos=0.3)
        
    cap_str = f" | Capacity Limit: {capacity:.1f}" if capacity else ""
    plt.title(f"{title}{cap_str}", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="png", dpi=150)
    plt.close()


def visualize_all_graphs():
    # Define the two source directories and their corresponding output folder names
    directories_to_process = [
        (GRAPHS_DIR, "graphs"),
        (GRAPHS_DIR.parent / "graphs_non_dag", "graphs_non_dag")
    ]
    
    for search_dir, folder_name in directories_to_process:
        if not search_dir.exists():
            print(f"Directory not found, skipping: {search_dir}")
            continue
            
        print(f"\nProcessing directory: {search_dir.name}...")
        for file_path in search_dir.glob("*.json"):
            with open(file_path, "r") as f:
                data = json.load(f)
                
            G = nx.DiGraph()
            for edge in data["edges"]:
                G.add_edge(edge["u"], edge["v"], cost=edge["cost"], resource=edge["resource"])
                
            plt.figure(figsize=(10, 8))
            
            pos = get_optimal_layout(G)
            
            node_colors = ['lightblue' if n not in [data["source"], data["target"]] else 'lightgreen' for n in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
            
            alpha = 0.8 if data["num_nodes"] <= 20 else 0.2
            nx.draw_networkx_edges(
                G, pos, arrowstyle="->", arrowsize=15, alpha=alpha, 
                edge_color="gray", connectionstyle="arc3,rad=0.1"
            )
            
            if data["num_nodes"] <= 20:
                edge_labels = {(u, v): f"C:{d['cost']:.1f}\nR:{d['resource']:.1f}" for u, v, d in G.edges(data=True)}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, label_pos=0.3)
                
            topo_type = "DAG" if nx.is_directed_acyclic_graph(G) else "Non-DAG"
            plt.title(f"Benchmark Graph ({topo_type}) | V={data['num_nodes']} | Capacity={data['capacity']:.2f}", fontsize=14)
            plt.axis("off")
            plt.tight_layout()
            
            # Route output to the separated subfolders
            out_path = FIGURES_DIR / folder_name / f"topo_{data['num_nodes']}.pdf"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            
            plt.savefig(out_path, format="pdf", dpi=300)
            plt.close()
            print(f"Saved visualization: {out_path.name} to figures/{folder_name}/")

if __name__ == "__main__":
    visualize_all_graphs()