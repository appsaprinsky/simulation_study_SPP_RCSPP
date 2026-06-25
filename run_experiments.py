import json
import time
import pandas as pd
import networkx as nx
from pathlib import Path
from config import GRAPHS_DIR, RESULTS_DIR, FIGURES_DIR, NUM_RUNS_DAG, NUM_RUNS_NON_DAG, PLOT_SOLUTIONS

from exact_shortest_path import ExactSolvers
from genetic_algorithm import GeneticAlgorithm
from particle_swarm import ParticleSwarmOptimization
from simulated_quantum_annealing import SQAOptimizer
from visualize_graphs import plot_solution
from non_dag_solvers import solve_non_dag_exact

def load_graph(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    G = nx.DiGraph()
    for e in data["edges"]:
        G.add_edge(e["u"], e["v"], cost=e["cost"], resource=e["resource"])
    return G, data["source"], data["target"], data["capacity"]

def main():
    batches = [
        {
            "input_dir": GRAPHS_DIR,
            "folder_name": "graphs",
            "num_runs": NUM_RUNS_DAG
        },
        {
            "input_dir": GRAPHS_DIR.parent / "graphs_non_dag",
            "folder_name": "graphs_non_dag",
            "num_runs": NUM_RUNS_NON_DAG
        }
    ]

    for batch in batches:
        search_dir = batch["input_dir"]
        folder_name = batch["folder_name"]
        total_runs = batch["num_runs"]

        if not search_dir.exists():
            print(f"\n[WARNING] Directory not found, skipping: {search_dir}")
            continue

        print(f"\n==================================================")
        print(f" STARTING BATCH: {folder_name.upper()} ({total_runs} runs per solver)")
        print(f"==================================================")

        current_results_dir = RESULTS_DIR / folder_name
        current_results_dir.mkdir(parents=True, exist_ok=True)
        
        current_solutions_dir = FIGURES_DIR / "solutions" / folder_name
        current_solutions_dir.mkdir(parents=True, exist_ok=True)
        
        debug_log_path = current_results_dir / "path_sequences_comparison.txt"
        with open(debug_log_path, "w") as log_f:
            log_f.write(f"=== MANUAL PATH SEQUENCES COMPARISON LOG ({folder_name.upper()}) ===\n\n")

        results = []
        graph_files = sorted(list(search_dir.glob("*.json")))
        
        for g_file in graph_files:
            print(f"\n--- Processing {g_file.name} ---")
            G, src, tgt, cap = load_graph(g_file)
            v_size = len(G.nodes())
            
            print("Running Exact Solvers...")
            if nx.is_directed_acyclic_graph(G):
                # for DAG 
                exact_spp_cost, exact_spp_path = ExactSolvers.solve_spp(G, src, tgt)
                exact_rcspp_cost, exact_rcspp_path = ExactSolvers.solve_rcspp(G, src, tgt, cap)
            else:
                # ONLY for non-DAGs
                exact_spp_cost, exact_spp_path = solve_non_dag_exact(G, src, tgt, None)
                exact_rcspp_cost, exact_rcspp_path = solve_non_dag_exact(G, src, tgt, cap)

            (current_solutions_dir / f"V{v_size}").mkdir(parents=True, exist_ok=True)
            if PLOT_SOLUTIONS:
                plot_solution(G, exact_spp_path, f"Exact SPP | V={v_size} | Cost={exact_spp_cost:.2f}", current_solutions_dir / f"V{v_size}/Exact_SPP.png")
                plot_solution(G, exact_rcspp_path, f"Exact RCSPP | V={v_size} | Cost={exact_rcspp_cost:.2f}", current_solutions_dir / f"V{v_size}/Exact_RCSPP.png", cap)

            with open(debug_log_path, "a") as log_f:
                log_f.write(f"\nGraph Size: {v_size} Nodes | File: {g_file.name}\n")
                log_f.write(f"  -> Exact SPP Path:   {exact_spp_path} (Cost: {exact_spp_cost:.2f})\n")
                log_f.write(f"  -> Exact RCSPP Path: {exact_rcspp_path} (Cost: {exact_rcspp_cost:.2f})\n")

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
                run_costs_feasible = []
                run_costs_all = []
                feasibility_count = 0
                start_time = time.time()
                
                # based on the specific batch's total_runs
                for run_idx in range(total_runs):
                    solver = SolverClass(G, src, tgt, capacity_val)
                    cost, path, is_feas = solver.run()
                    
                    # 1. save the cost to the 'all' tracker, even if it is float('inf')
                    run_costs_all.append(cost)
                    
                    # 2. log record of the failure/infeasible path
                    with open(debug_log_path, "a") as log_f:
                        log_f.write(f"  [{prob_type}] {alg_name} Run {run_idx:02d} | Feasible={str(is_feas):5s} | Cost={cost:9.2f} | Path: {path}\n")
                    
                    # 3. attempt to plot if the algorithm actually returned a path sequence
                    if PLOT_SOLUTIONS and path:
                        out_png = current_solutions_dir / f"V{v_size}" / f"{alg_name}_{prob_type}_run{run_idx}.png"
                        title = f"{alg_name} {prob_type} | Run {run_idx} | Cost={cost:.2f} | Feasible={is_feas}"
                        plot_solution(G, path, title, out_png, capacity_val)
                    
                    # 4. filter for strict feasibility
                    if is_feas and cost != float('inf'):
                        run_costs_feasible.append(cost)
                        feasibility_count += 1
                        
                total_time = time.time() - start_time
                tts = total_time / total_runs
                rho = (feasibility_count / total_runs) * 100
                
                if run_costs_feasible:
                    f_best_feas = min(run_costs_feasible)
                    f_mean_feas = sum(run_costs_feasible) / len(run_costs_feasible)
                    sigma = pd.Series(run_costs_feasible).std() if len(run_costs_feasible) > 1 else 0.0
                    gap = ((f_mean_feas - exact_cost) / abs(exact_cost)) * 100 if exact_cost != 0 else 0
                else:
                    f_best_feas = f_mean_feas = sigma = gap = None

                if run_costs_all:
                    f_best_all = min(run_costs_all)
                    f_mean_all = sum(run_costs_all) / len(run_costs_all)
                else:
                    f_best_all = f_mean_all = None

                results.append({
                    "Nodes": v_size,
                    "Problem": prob_type,
                    "Algorithm": alg_name,
                    "Exact_Cost": exact_cost,
                    "TTS_sec": tts,
                    "Feasibility_Rho": rho,
                    "f_best": f_best_feas,
                    "f_mean": f_mean_feas,
                    "best_all": f_best_all,
                    "mean_all": f_mean_all,
                    "Sigma": sigma,
                    "Optimality_Gap": gap
                })

        df = pd.DataFrame(results)
        df.to_csv(current_results_dir / "benchmark_results.csv", index=False)
        df.to_excel(current_results_dir / "benchmark_results.xlsx", index=False)
        
        print(f"\n--- Batch {folder_name.upper()} Complete ---")
        print(f"Results saved to: {current_results_dir}/")

if __name__ == "__main__":
    main()