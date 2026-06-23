import pulp
import networkx as nx
from typing import Tuple, List

def solve_non_dag_exact(G: nx.DiGraph, source: int, target: int, capacity: float = None) -> Tuple[float, List[int]]:
    prob = pulp.LpProblem("Routing", pulp.LpMinimize)
    x = {(u, v): pulp.LpVariable(f"x_{u}_{v}", cat=pulp.LpBinary) for u, v in G.edges()}
    
    prob += pulp.lpSum([G[u][v]['cost'] * x[(u, v)] for u, v in G.edges()])
    
    if capacity is not None:
        prob += pulp.lpSum([G[u][v]['resource'] * x[(u, v)] for u, v in G.edges()]) <= capacity
        
    for node in G.nodes():
        in_flow = pulp.lpSum([x[(u, node)] for u, _ in G.in_edges(node)])
        out_flow = pulp.lpSum([x[(node, v)] for _, v in G.out_edges(node)])
        if node == source: prob += (out_flow - in_flow == 1)
        elif node == target: prob += (in_flow - out_flow == 1)
        else: 
            prob += (in_flow == out_flow)
            prob += (in_flow <= 1)

    # Iterative Lazy Constraint loop
    for _ in range(20):
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=5)
        status = prob.solve(solver)
        
        if pulp.LpStatus[status] != 'Optimal': break
        
        active_edges = [(u, v) for (u, v), var in x.items() if pulp.value(var) > 0.5]
        subgraph = nx.DiGraph(active_edges)
        
        try:
            # find_cycle returns list of (u, v) tuples
            cycle = list(nx.find_cycle(subgraph, orientation="original"))
            # Just extract the u, v parts in case the tuple has extra data
            cycle_edges = [(u, v) for u, v, *data in cycle]
            
            # Constraint: Sum of edges in cycle must be < len(cycle)
            prob += pulp.lpSum([x[(u, v)] for u, v in cycle_edges]) <= len(cycle_edges) - 1
        except nx.NetworkXNoCycle:
            break
            
    # Reconstruction
    if pulp.LpStatus[prob.solve(solver)] == 'Optimal':
        cost = pulp.value(prob.objective)
        path = [source]; curr = source
        while curr != target:
            next_node = None
            for v in G.successors(curr):
                if pulp.value(x.get((curr, v), 0)) and pulp.value(x[(curr, v)]) > 0.5:
                    next_node = v
                    break
            if next_node is None: break
            path.append(next_node); curr = next_node
        return cost, path
    return float('inf'), []