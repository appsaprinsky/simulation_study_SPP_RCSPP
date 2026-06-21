import json
import time
import pandas as pd
import networkx as nx
from pathlib import Path
from config import GRAPHS_DIR, RESULTS_DIR, FIGURES_DIR, NUM_RUNS

from exact_shortest_path import ExactSolvers
from genetic_algorithm import GeneticAlgorithm
from particle_swarm import ParticleSwarmOptimization
from simulated_quantum_annealing import SQAOptimizer
from visualize_graphs import plot_solution

def load_graph(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    G = nx.DiGraph()
    for e in data["edges"]:
        G.add_edge(e["u"], e["v"], cost=e["cost"], resource=e["resource"])
    return G, data["source"], data["target"], data["capacity"]

def main():
    results = []
    graph_files = list(GRAPHS_DIR.glob("*.json"))
    
    # Base directory for solution plots
    solutions_dir = FIGURES_DIR / "solutions"
    solutions_dir.mkdir(exist_ok=True)
    
    for g_file in graph_files:
        print(f"\n--- Processing {g_file.name} ---")
        G, src, tgt, cap = load_graph(g_file)
        v_size = len(G.nodes())
        
        print("Running Exact Solvers...")
        exact_spp_cost, exact_spp_path = ExactSolvers.solve_spp(G, src, tgt)
        exact_rcspp_cost, exact_rcspp_path = ExactSolvers.solve_rcspp(G, src, tgt, cap)
        
        # Save exact baselines visuals
        plot_solution(G, exact_spp_path, f"Exact SPP | V={v_size} | Cost={exact_spp_cost:.2f}", solutions_dir / f"V{v_size}/Exact_SPP.png")
        plot_solution(G, exact_rcspp_path, f"Exact RCSPP | V={v_size} | Cost={exact_rcspp_cost:.2f}", solutions_dir / f"V{v_size}/Exact_RCSPP.png", cap)

        tasks = [
            ("SPP", "GA", GeneticAlgorithm, exact_spp_cost, None),
            ("RCSPP", "GA", GeneticAlgorithm, exact_rcspp_cost, cap),
            ("SPP", "PSO", ParticleSwarmOptimization, exact_spp_cost, None),
            ("RCSPP", "PSO", ParticleSwarmOptimization, exact_rcspp_cost, cap),
            ("SPP", "SQA", SQAOptimizer, exact_spp_cost, None),
            ("RCSPP", "SQA", SQAOptimizer, exact_rcspp_cost, cap)
        ]
        
        for prob_type, alg_name, SolverClass, exact_cost, capacity_val in tasks:
            print(f"Executing {alg_name} for {prob_type}...")
            run_costs = []
            feasibility_count = 0
            start_time = time.time()
            
            for run_idx in range(NUM_RUNS):
                solver = SolverClass(G, src, tgt, capacity_val)
                cost, path, is_feas = solver.run()
                
                if is_feas and cost != float('inf'):
                    run_costs.append(cost)
                    feasibility_count += 1
                    
                    # Generate visualization for this run
                    out_png = solutions_dir / f"V{v_size}" / f"{alg_name}_{prob_type}_run{run_idx}.png"
                    title = f"{alg_name} {prob_type} | V={v_size} | Run {run_idx} | Cost={cost:.2f}"
                    plot_solution(G, path, title, out_png, capacity_val)
                    
            total_time = time.time() - start_time
            tts = total_time / NUM_RUNS
            
            rho = (feasibility_count / NUM_RUNS) * 100
            
            if run_costs:
                f_best = min(run_costs)
                f_mean = sum(run_costs) / len(run_costs)
                sigma = pd.Series(run_costs).std() if len(run_costs) > 1 else 0.0
                gap = ((f_mean - exact_cost) / abs(exact_cost)) * 100 if exact_cost != 0 else 0
            else:
                f_best = f_mean = sigma = gap = None

            results.append({
                "Nodes": v_size,
                "Problem": prob_type,
                "Algorithm": alg_name,
                "Exact_Cost": exact_cost,
                "TTS_sec": tts,
                "Feasibility_Rho": rho,
                "f_best": f_best,
                "f_mean": f_mean,
                "Sigma": sigma,
                "Optimality_Gap": gap
            })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    df.to_excel(RESULTS_DIR / "benchmark_results.xlsx", index=False)
    print("\nExperiments Complete. Data and visualizations saved.")

if __name__ == "__main__":
    main()