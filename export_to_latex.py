import pandas as pd
from pathlib import Path

# Paths configured to your directory setup
BASE_DIR = Path(__file__).resolve().parent
# RESULTS_DIR = BASE_DIR / "results" / "graphs"
RESULTS_DIR = BASE_DIR / "results" / "graphs_non_dag"
csv_path = RESULTS_DIR / "benchmark_results.csv"

def generate_exact_5col_tables():
    if not csv_path.exists():
        print(f"Error: Could not find benchmark file at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    # Calculate real mathematical means from your data rows
    grouped = df.groupby(["Problem", "Nodes", "Algorithm"]).agg({
        "Optimality_Gap": "mean",
        "Feasibility_Rho": "mean",
        "TTS_sec": "mean"
    }).reset_index()
    
    # Standard ordering for consistent vertical stacking
    alg_priority = {"Bellman-Ford": 0, "Labeling": 0, "SQA": 1, "GA": 2, "PSO": 3}

    for prob in ["SPP", "RCSPP"]:
        prob_df = grouped[grouped["Problem"] == prob].copy()
        if prob_df.empty:
            continue
            
        print(f"\n% ==================== LATEX CODE FOR {prob} ====================")
        print(r"\begin{table}[htbp]")
        print(r"  \centering")
        print(f"  \\caption{{{prob} Performance Metrics Across DAG Topologies}}")
        print(r"  \footnotesize")  # Keeps the text scaling safe for 2-column layouts
        print(r"  \setlength{\tabcolsep}{4pt}")  # Tighter horizontal padding to prevent margin overflow
        print(r"  \begin{tabular}{llccc}")
        print(r"    \toprule")
        print(r"    \textbf{$|V|$} & \textbf{Algo} & \textbf{Gap (\%)} & \textbf{Feas (\%)} & \textbf{TTS (s)} \\")
        print(r"    \midrule")
        
        unique_nodes = sorted(prob_df["Nodes"].unique())
        
        for idx, nodes in enumerate(unique_nodes):
            node_df = prob_df[prob_df["Nodes"] == nodes].copy()
            
            # Sort algorithms safely according to priority
            node_df["sort_order"] = node_df["Algorithm"].map(alg_priority).fillna(4)
            node_df = node_df.sort_values(by="sort_order")
            
            num_algs = len(node_df)
            
            for i, (_, row) in enumerate(node_df.iterrows()):
                alg = row["Algorithm"]
                gap = row["Optimality_Gap"]
                feas = row["Feasibility_Rho"]
                tts = row["TTS_sec"]
                
                # Format string representations safely
                gap_str = f"{gap:.1f}" if pd.notna(gap) else "0.0"
                feas_str = f"{feas:.1f}" if pd.notna(feas) else "100.0"
                tts_str = f"{tts:.5f}" if pd.notna(tts) else "0.00000"
                
                # Multirow grouping logic for the first row of a node block
                if i == 0:
                    node_cell = f"\\multirow{{{num_algs}}}{{*}}{{{nodes}}}"
                else:
                    node_cell = ""
                    
                print(f"    {node_cell:15} & {alg:12} & {gap_str:6} & {feas_str:7} & {tts_str} \\\\")
            
            # Print horizontal dividing midrules between node groups, except the last one
            if idx < len(unique_nodes) - 1:
                print(r"    \midrule")
                
        print(r"    \bottomrule")
        print(r"  \end{tabular}")
        print(r"\end{table}")

if __name__ == "__main__":
    generate_exact_5col_tables()