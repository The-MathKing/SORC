import numpy as np
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import wasserstein_distance
import time

def compute_sorc(G, k=10, alpha=0.5):
    """
    Computes Sliced Ollivier-Ricci Curvature (SORC) for all edges in G.
    
    Parameters:
    - G: networkx.Graph (undirected, unweighted)
    - k: number of Laplacian eigenvectors to use
    - alpha: idleness parameter for the lazy random walk
    
    Returns:
    - edge_sorc: dict mapping edge (u, v) to its SORC value
    - runtime: float, time taken to compute
    """
    start_time = time.time()
    
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    
    # 1. Laplacian Eigendecomposition
    A = nx.adjacency_matrix(G, nodelist=nodes)
    degrees = np.array(A.sum(axis=1)).flatten()
    D = sp.diags(degrees)
    L = D - A
    
    # Compute bottom k+1 eigenvectors (using shift-invert mode for smallest eigenvalues)
    # sigma=-1e-3 helps find eigenvalues near 0 robustly
    num_eigs = min(k + 1, n - 1)
    vals, vecs = eigsh(L.astype(float), k=num_eigs, which='SM', sigma=-1e-3)
    
    # Sort eigenvalues and discard the first trivial one (lambda=0)
    idx_sort = np.argsort(vals)
    vecs = vecs[:, idx_sort[1:]]
    
    # 2. Lipschitz Normalization
    # For each eigenvector, we find the maximum absolute difference across all edges
    edges = list(G.edges())
    if len(edges) == 0:
        return {}, time.time() - start_time
        
    idx_u = np.array([node_to_idx[u] for u, v in edges])
    idx_v = np.array([node_to_idx[v] for u, v in edges])
    
    # Edge differences shape: (num_edges, k)
    edge_diffs = np.abs(vecs[idx_u, :] - vecs[idx_v, :])
    max_diffs = np.max(edge_diffs, axis=0) # shape: (k,)
    
    # Avoid division by zero for any disconnected components / zero vectors
    max_diffs[max_diffs < 1e-12] = 1.0 
    
    # f_i are the Lipschitz-normalized eigenvectors
    F = vecs / max_diffs
    
    # 3. Precompute Local Random Walk Measures Support
    # For each node u, m_u is supported on {u} U N(u)
    node_supports = []
    node_weights = []
    
    for u in nodes:
        neighbors = list(G.neighbors(u))
        deg_u = len(neighbors)
        if deg_u > 0:
            support = [node_to_idx[u]] + [node_to_idx[v] for v in neighbors]
            weights = [alpha] + [(1 - alpha) / deg_u] * deg_u
        else:
            support = [node_to_idx[u]]
            weights = [1.0]
        node_supports.append((np.array(support), np.array(weights)))
        
    # 4. Compute 1D Wasserstein over edges and aggregate
    edge_sorc = {}
    for u, v in edges:
        idx_u_node = node_to_idx[u]
        idx_v_node = node_to_idx[v]
        
        supp_u, w_u = node_supports[idx_u_node]
        supp_v, w_v = node_supports[idx_v_node]
        
        max_w1 = 0.0
        
        # Project onto each normalized eigenvector
        for i in range(F.shape[1]):
            # 1D coordinates of the support
            u_vals = F[supp_u, i]
            v_vals = F[supp_v, i]
            
            # Compute 1D Wasserstein distance
            w1 = wasserstein_distance(u_vals, v_vals, w_u, w_v)
            if w1 > max_w1:
                max_w1 = w1
                
        # k_SORC = 1 - W1 / d(u,v), where d(u,v) = 1 for edges
        edge_sorc[(u, v)] = 1.0 - max_w1
        edge_sorc[(v, u)] = edge_sorc[(u, v)]
        
    runtime = time.time() - start_time
    return edge_sorc, runtime

if __name__ == "__main__":
    # Quick test on a barbell graph
    print("Testing SORC on Barbell Graph...")
    G = nx.barbell_graph(10, 2)
    sorc_dict, t = compute_sorc(G, k=5)
    print(f"Computed SORC for {len(G.edges())} edges in {t:.4f} seconds.")
    
    # Check the bridge edge (connecting the two bells via the path)
    # For barbell(10, 2), nodes 0-9 are bell 1, 10-11 are path, 12-21 are bell 2
    # The path edges are (9, 10), (10, 11), (11, 12)
    print(f"SORC of bridge edge (9, 10): {sorc_dict[(9, 10)]:.4f} (Expected: highly negative)")
    print(f"SORC of clique edge (0, 1): {sorc_dict[(0, 1)]:.4f} (Expected: positive)")
