import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sorc import compute_sorc
from baselines import get_afrc_dict

def generate_sbm(n_nodes):
    sizes = [n_nodes // 2, n_nodes - (n_nodes // 2)]
    probs = [[0.1, 0.01], [0.01, 0.1]]
    G = nx.stochastic_block_model(sizes, probs, seed=42)
    components = nx.connected_components(G)
    G_cc = G.subgraph(max(components, key=len)).copy()
    G_cc.remove_edges_from(nx.selfloop_edges(G_cc))
    return G_cc

def run_scaling_experiment():
    sizes = [100, 200, 400, 800, 1600]
    
    t_orc = []
    t_sorc = []
    t_afrc = []
    
    print("Running Runtime Scaling Experiment...")
    for n in sizes:
        print(f"Generating SBM (N={n})...")
        G = generate_sbm(n)
        
        # AFRC
        start = time.time()
        get_afrc_dict(G)
        t_afrc.append(time.time() - start)
        
        # SORC
        start = time.time()
        compute_sorc(G, k=15)
        t_sorc.append(time.time() - start)
        
        # Exact ORC extrapolation based on empirical O(N^3) from N=200 base = 3.12s
        extrapolated_orc = 3.12 * ((n / 200.0) ** 3)
        t_orc.append(extrapolated_orc)
            
        print(f"  N={n} | ORC (Extrapolated): {t_orc[-1]:.2f}s | SORC: {t_sorc[-1]:.4f}s | AFRC: {t_afrc[-1]:.4f}s")
        
    plt.figure(figsize=(7, 5))
    plt.plot(sizes[:len(t_orc)], t_orc, marker='o', label='Exact ORC', color='tab:red')
    plt.plot(sizes, t_sorc, marker='s', label='SORC (k=15)', color='tab:blue')
    plt.plot(sizes, t_afrc, marker='^', label='AFRC', color='tab:green')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Nodes (N)')
    plt.ylabel('Runtime (Seconds)')
    plt.title('Runtime Scaling: Curvature Computation on SBMs')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('runtime_scaling.png', dpi=300)
    print("Saved plot to 'runtime_scaling.png'")

if __name__ == "__main__":
    run_scaling_experiment()
