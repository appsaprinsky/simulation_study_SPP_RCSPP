import json
import networkx as nx
import matplotlib.pyplot as plt
from config import GRAPHS_DIR, FIGURES_DIR

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