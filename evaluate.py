import numpy as np
import networkx as nx
import time
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sorc import compute_sorc

def generate_sbm(n_blocks=2, block_size=100, p_in=0.2, p_out=0.01, seed=42):
    """
    Generate a Stochastic Block Model graph.
    """
    sizes = [block_size] * n_blocks
    probs = np.full((n_blocks, n_blocks), p_out)
    np.fill_diagonal(probs, p_in)
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    # Extract largest connected component to ensure eigenvalues are well-behaved
    components = nx.connected_components(G)
    largest_cc = max(components, key=len)
    G_cc = G.subgraph(largest_cc).copy()
    
    # Remove self-loops if any
    G_cc.remove_edges_from(nx.selfloop_edges(G_cc))
    return G_cc

def evaluate_sorc():
    print("Generating SBM test graph (~200 nodes)...")
    G = generate_sbm(n_blocks=2, block_size=100, p_in=0.2, p_out=0.01)
    
    print(f"Graph properties: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")
    
    # 1. Compute Exact ORC
    print("Computing exact ORC (this may take a moment)...")
    start_orc = time.time()
    orc = OllivierRicci(G, alpha=0.5, verbose="ERROR")
    orc.compute_ricci_curvature()
    time_orc = time.time() - start_orc
    
    # 2. Compute SORC
    print("Computing SORC (k=20)...")
    sorc_dict, time_sorc = compute_sorc(G, k=20, alpha=0.5)
    
    print(f"\n--- Runtimes ---")
    print(f"Exact ORC: {time_orc:.4f}s")
    print(f"SORC:      {time_sorc:.4f}s")
    
    # 3. Correlation Analysis
    edges = list(G.edges())
    orc_vals = []
    sorc_vals = []
    
    for u, v in edges:
        # GraphRicciCurvature stores edge curvature as edge attribute
        orc_val = G[u][v].get('ricciCurvature', 0.0)
        orc_vals.append(orc_val)
        
        # Look up SORC, sorting the tuple just in case
        sorc_val = sorc_dict.get((u, v), sorc_dict.get((v, u), 0.0))
        sorc_vals.append(sorc_val)
        
    orc_vals = np.array(orc_vals)
    sorc_vals = np.array(sorc_vals)
    
    corr, pval = spearmanr(orc_vals, sorc_vals)
    print(f"\n--- Correlation ---")
    print(f"Spearman correlation between exact ORC and SORC: {corr:.4f} (p-value: {pval:.2e})")
    
    # Generate scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(orc_vals, sorc_vals, alpha=0.5, s=15, color='blue')
    plt.xlabel('Exact Ollivier-Ricci Curvature (ORC)')
    plt.ylabel('Sliced Ollivier-Ricci Curvature (SORC)')
    plt.title(f'SORC vs Exact ORC on SBM (Spearman r = {corr:.3f})')
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig('sorc_vs_orc_correlation.png', dpi=300, bbox_inches='tight')
    print("Saved correlation plot to 'sorc_vs_orc_correlation.png'")
    
    # Identify bridge edges vs community edges for validation
    bridge_sorc = []
    comm_sorc = []
    
    for u, v in edges:
        if (u < 100 and v >= 100) or (v < 100 and u >= 100):
            bridge_sorc.append(sorc_dict.get((u,v), sorc_dict.get((v,u))))
        else:
            comm_sorc.append(sorc_dict.get((u,v), sorc_dict.get((v,u))))
            
    print(f"\n--- Topological Distribution ---")
    print(f"Average SORC for bridge/inter-community edges: {np.mean(bridge_sorc):.4f}")
    print(f"Average SORC for intra-community edges:        {np.mean(comm_sorc):.4f}")

if __name__ == "__main__":
    evaluate_sorc()
