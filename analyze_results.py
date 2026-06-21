import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import RESULTS_DIR, FIGURES_DIR

def run_analysis():
    csv_path = RESULTS_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        print("No results file found.")
        return
        
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid")
    
    for prob in ["SPP", "RCSPP"]:
        prob_df = df[df["Problem"] == prob]
        
        # 1. Optimality Gap Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=prob_df, x="Nodes", y="Optimality_Gap", hue="Algorithm")
        plt.title(f"Optimality Gap ($\Delta$) vs Graph Size - {prob}")
        plt.ylabel("Gap (%)")
        plt.savefig(FIGURES_DIR / f"{prob}_Optimality_Gap.png", dpi=300)
        plt.close()
        
        # 2. Time To Solution (TTS) Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=prob_df, x="Nodes", y="TTS_sec", hue="Algorithm", marker="o")
        plt.yscale("log")
        plt.title(f"Mean Time To Solution (TTS) - {prob}")
        plt.ylabel("Time (Seconds - Log Scale)")
        plt.savefig(FIGURES_DIR / f"{prob}_TTS.png", dpi=300)
        plt.close()
        
        # 3. Feasibility Rate (rho) Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=prob_df, x="Nodes", y="Feasibility_Rho", hue="Algorithm")
        plt.title(f"Feasibility Success Rate ($\\rho$) - {prob}")
        plt.ylabel("Feasible Solutions (%)")
        plt.ylim(0, 105)
        plt.savefig(FIGURES_DIR / f"{prob}_Feasibility.png", dpi=300)
        plt.close()
        
    print(f"Analytics generated. Check the '{FIGURES_DIR.name}' directory.")

if __name__ == "__main__":
    run_analysis()