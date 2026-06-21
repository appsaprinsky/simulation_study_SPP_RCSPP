import time
import random
import numpy as np
import networkx as nx
from dwave.samplers import SimulatedAnnealingSampler
import collections

# ==========================================
# 1. GRAPH GENERATION & INSTANCE CREATION
# ==========================================
def generate_instance(num_nodes, edge_prob=0.4):
    """
    Generates a directed graph with edge costs, negative node weights, 
    and edge resource consumption.
    """
    G = nx.gnp_random_graph(num_nodes, edge_prob, directed=True)
    # Ensure graph is connected sequentially to allow a valid path
    for i in range(num_nodes - 1):
        G.add_edge(i, i + 1)
        
    random.seed(42)
    
    # Assign attributes
    for u, v in G.edges():
        G[u][v]['cost'] = random.randint(2, 15)
        G[u][v]['resource'] = random.randint(1, 5)
        
    for node in G.nodes():
        # Source and sink have 0 node weight for simplicity, others can be negative
        if node in [0, num_nodes - 1]:
            G.nodes[node]['weight'] = 0
        else:
            G.nodes[node]['weight'] = random.randint(-10, -1)
            
    # Max allowed resource (Capacity)
    max_capacity = int(num_nodes * 2.5)
    return G, 0, num_nodes - 1, max_capacity

# ==========================================
# 2. CLASSICAL BASELINE: LABELING ALGORITHM
# ==========================================
def labeling_algorithm(G, source, target, max_capacity):
    """
    An exact Labeling Algorithm for the Capacitated Shortest Path Problem.
    Labels store: (cost, resource_used)
    """
    # labels[node] is a list of tuples (cost, resource)
    labels = collections.defaultdict(list)
    labels[source].append((0, 0))
    
    # Queue for processing nodes
    queue = collections.deque([source])
    
    while queue:
        u = queue.popleft()
        for v in G.successors(u):
            edge_cost = G[u][v]['cost']
            edge_res = G[u][v]['resource']
            node_weight = G.nodes[v]['weight']
            
            updated = False
            for c_old, r_old in labels[u]:
                c_new = c_old + edge_cost + node_weight
                r_new = r_old + edge_res
                
                if r_new <= max_capacity:
                    # Check dominance: discard if dominated by an existing label
                    dominated = False
                    for c_exist, r_exist in labels[v]:
                        if c_exist <= c_new and r_exist <= r_new:
                            dominated = True
                            break
                    
                    if not dominated:
                        # Remove existing labels dominated by the new label
                        labels[v] = [lbl for lbl in labels[v] if not (c_new <= lbl[0] and r_new <= lbl[1])]
                        labels[v].append((c_new, r_new))
                        updated = True
            
            if updated and v not in queue:
                queue.append(v)
                
    if not labels[target]:
        return float('inf'), None
        
    best_cost = min(c for c, r in labels[target])
    return best_cost, None

# ==========================================
# 3. QUANTUM SIMULATOR (QUBO FORMULATION)
# ==========================================
def solve_with_qubo_simulator(G, source, target, max_capacity):
    """
    Formulates the CSPP into a QUBO and solves it using D-Wave's 
    Simulated Annealing Sampler.
    """
    edges = list(G.edges())
    num_edges = len(edges)
    
    # Map edges to binary variable indices
    edge_to_idx = {edge: i for i, edge in enumerate(edges)}
    
    # Slack variables needed to handle the capacity inequality constraint:
    # sum(res_e * x_e) <= max_capacity -> sum(res_e * x_e) + sum(2^k * s_k) = max_capacity
    num_slacks = int(np.log2(max_capacity)) + 1
    total_vars = num_edges + num_slacks
    
    # Initialize QUBO Matrix
    Q = np.zeros((total_vars, total_vars))
    
    # Penalty Multipliers
    P_flow = 150.0      # Penalty for path discontinuity
    P_cap = 50.0        # Penalty for breaking resource limits
    
    # 1. Objective Function (Minimize Edge costs + Node Weights)
    for edge, idx in edge_to_idx.items():
        u, v = edge
        # Cost is edge cost + destination node weight modification
        total_e_cost = G[u][v]['cost'] + G.nodes[v]['weight']
        Q[idx, idx] += total_e_cost

    # 2. Flow Conservation Constraints: (Inflow - Outflow = b_i)^2
    for node in G.nodes():
        if node == source:
            b = 1
        elif node == target:
            b = -1
        else:
            b = 0
            
        # Equation: ( sum(x_in) - sum(x_out) - b )^2
        # = (sum(x_in))^2 + (sum(x_out))^2 + b^2 - 2*b*sum(x_in) + 2*b*sum(x_out) - 2*sum(x_in)*sum(x_out)
        in_edges = [edge_to_idx[(u, v)] for u, v in G.in_edges(node)]
        out_edges = [edge_to_idx[(u, v)] for u, v in G.out_edges(node)]
        
        # Linear terms
        for idx in in_edges:
            Q[idx, idx] -= 2 * b * P_flow
        for idx in out_edges:
            Q[idx, idx] += 2 * b * P_flow
            
        # Quadratic terms
        all_flow_vars = in_edges + out_edges
        signs = {idx: 1 for idx in in_edges}
        for idx in out_edges:
            signs[idx] = -1
            
        for i in range(len(all_flow_vars)):
            for j in range(len(all_flow_vars)):
                idx_i = all_flow_vars[i]
                idx_j = all_flow_vars[j]
                val = P_flow * signs[idx_i] * signs[idx_j]
                if idx_i <= idx_j:
                    Q[idx_i, idx_j] += val
                else:
                    Q[idx_j, idx_i] += val

    # 3. Capacity Constraint (Inequality converted to equality via slacks)
    # ( sum(res_e * x_e) + sum(2^k * s_k) - max_capacity )^2
    slack_indices = [num_edges + k for k in range(num_slacks)]
    slack_weights = [2**k for k in range(num_slacks)]
    
    # Combine structural and slack coefficients for easy matrix building
    cap_vars = []
    cap_weights = []
    for edge, idx in edge_to_idx.items():
        cap_vars.append(idx)
        cap_weights.append(G[edge[0]][edge[1]]['resource'])
    for idx, w in zip(slack_indices, slack_weights):
        cap_vars.append(idx)
        cap_weights.append(w)
        
    for i in range(len(cap_vars)):
        idx_i = cap_vars[i]
        w_i = cap_weights[i]
        # Linear offset from expansion: -2 * max_capacity * w_i
        Q[idx_i, idx_i] -= 2 * max_capacity * w_i * P_cap
        
        for j in range(len(cap_vars)):
            idx_j = cap_vars[j]
            w_j = cap_weights[j]
            val = P_cap * w_i * w_j
            if idx_i <= idx_j:
                Q[idx_i, idx_j] += val
            else:
                Q[idx_j, idx_i] += val

    # Convert dictionary layout for D-Wave format
    qubo_dict = {}
    for i in range(total_vars):
        for j in range(i, total_vars):
            if Q[i, j] != 0:
                qubo_dict[(i, j)] = Q[i, j]

    # Sample using D-Wave Ocean's simulated annealing tool
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(qubo_dict, num_reads=100, num_sweeps=2000)
    
    best_sample = sampleset.first.sample
    best_energy = sampleset.first.energy
    
    # Parse the chosen path cost from binary outputs
    sim_cost = 0
    chosen_edges = []
    for edge, idx in edge_to_idx.items():
        if best_sample[idx] == 1:
            u, v = edge
            sim_cost += G[u][v]['cost'] + G.nodes[v]['weight']
            chosen_edges.append(edge)
            
    return sim_cost, chosen_edges

# ==========================================
# 4. EXECUTION AND COMPARISON BENCHMARK
# ==========================================
sizes = [5, 10, 20, 50, 100, 200]
print(f"{'Nodes':<8}{'Label Cost':<12}{'Label Time(s)':<15}{'QUBO Cost':<12}{'QUBO Time(s)':<15}")
print("-" * 65)

for n in sizes:
    # Use smaller edge probabilities for massive graphs to keep variables bounded
    prob = 0.5 if n <= 20 else 0.1
    G, src, tgt, cap = generate_instance(n, edge_prob=prob)
    
    # Benchmark Classical Labeling
    t0 = time.time()
    lbl_cost, _ = labeling_algorithm(G, src, tgt, cap)
    t_lbl = time.time() - t0
    
    # Benchmark Quantum Annealer (Simulator)
    # We stop QUBO for sizes 100+ in this local code structure because the variable matrix grows 
    # quadratically (~O(E^2)), highlighting classical scaling bottlenecks for QUBO setup.
    if n <= 50:
        t1 = time.time()
        qubo_cost, _ = solve_with_qubo_simulator(G, src, tgt, cap)
        t_qubo = time.time() - t1
        qubo_cost_str = f"{qubo_cost:.1f}"
        t_qubo_str = f"{t_qubo:.4f}"
    else:
        qubo_cost_str = "Timeout/OOM"
        t_qubo_str = "N/A"
        
    lbl_cost_str = f"{lbl_cost:.1f}" if lbl_cost != float('inf') else "No Path"
    print(f"{n:<8}{lbl_cost_str:<12}{t_lbl:<15.4f}{qubo_cost_str:<12}{t_qubo_str:<15}")