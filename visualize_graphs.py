import json
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from config import GRAPHS_DIR, FIGURES_DIR

def plot_solution(G: nx.DiGraph, path: list, title: str, out_path: Path, capacity: float = None):
    """Overlays the selected solution path onto the graph topology."""
    plt.figure(figsize=(10, 8))
    
    # Use topological sort for layout since we are using DAGs
    try:
        layers = list(nx.topological_generations(G))
        # Assign layers FIRST
        for layer, nodes in enumerate(layers):
            for node in nodes:
                G.nodes[node]["layer"] = layer
                
        # THEN call the layout generator
        pos = nx.multipartite_layout(G, subset_key="layer")
    except nx.NetworkXUnfeasible:
        # Fallback to Kamada-Kawai if graph is somehow not a perfect DAG
        pos = nx.kamada_kawai_layout(G)
    
    # Draw base nodes
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=300, edgecolors='black')
    
    # Highlight solution nodes
    if path:
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='lightgreen', node_size=400, edgecolors='black')
        
    # Draw base edges
    nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=10, alpha=0.3, edge_color="gray")
    
    # Highlight solution edges
    path_edges = list(zip(path[:-1], path[1:])) if path else []
    if path_edges:
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2.5, arrowstyle="->", arrowsize=15)
        
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    
    # Render labels only for smaller graphs to avoid visual clutter
    if len(G.nodes()) <= 20:
        edge_labels = {(u, v): f"C:{d['cost']:.1f}" for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        
    cap_str = f" | Capacity Limit: {capacity:.1f}" if capacity else ""
    plt.title(f"{title}{cap_str}", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    
    # Ensure directory exists before saving
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, format="png", dpi=150)
    plt.close()


def visualize_all_graphs():
    for file_path in GRAPHS_DIR.glob("*.json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            
        G = nx.DiGraph()
        for edge in data["edges"]:
            G.add_edge(edge["u"], edge["v"], cost=edge["cost"], resource=edge["resource"])
            
        plt.figure(figsize=(10, 8))
        
        # Use Kamada-Kawai layout for aesthetically pleasing distance plotting
        pos = nx.kamada_kawai_layout(G)
        
        # Draw Nodes
        node_colors = ['lightblue' if n not in [data["source"], data["target"]] else 'lightgreen' for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
        
        # Draw Edges (with alpha to manage dense graphs)
        alpha = 0.8 if data["num_nodes"] <= 20 else 0.2
        nx.draw_networkx_edges(G, pos, arrowstyle="->", arrowsize=15, alpha=alpha, edge_color="gray")
        
        # Edge Labels (Only for small graphs to avoid clutter)
        if data["num_nodes"] <= 20:
            edge_labels = {(u, v): f"C:{d['cost']:.1f}\nR:{d['resource']:.1f}" for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
            
        plt.title(f"Benchmark Graph | V={data['num_nodes']} | Capacity={data['capacity']:.2f}", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        
        out_path = FIGURES_DIR / f"topo_{data['num_nodes']}.pdf"
        plt.savefig(out_path, format="pdf", dpi=300)
        plt.close()
        print(f"Saved visualization: {out_path.name}")

if __name__ == "__main__":
    visualize_all_graphs()