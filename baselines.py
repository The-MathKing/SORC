import numpy as np
import scipy.sparse as sp
import networkx as nx

def compute_afrc(A: sp.csr_matrix, gamma=3.0) -> np.ndarray:
    """
    Computes Augmented Forman-Ricci Curvature (AFRC) for all edges in the sparse adjacency matrix A.
    Returns an array of F* values corresponding to the nonzero edges in A.
    """
    n = A.shape[0]
    A_sq = A.dot(A)
    u, v = A.nonzero()
    edge_weights = np.array(A[u, v]).flatten()
    
    d = np.array(np.abs(A).sum(axis=1)).flatten()
    du = d[u]
    dv = d[v]
    
    A3_uv = np.zeros(len(u))
    for i in range(len(u)):
        row = u[i]
        col = v[i]
        A3_uv[i] = A[row, :].dot(A_sq[:, col]).sum()
        
    C4_sum = edge_weights * A3_uv - du - dv + 1
    max_bound = np.maximum(1, (du - 1) * (dv - 1))
    normalized_term = C4_sum / max_bound
    
    F_star = 4 - du - dv + gamma * np.minimum(du - 1, dv - 1) * normalized_term
    return F_star

def get_afrc_dict(G: nx.Graph, gamma=3.0):
    """
    Convenience wrapper to return AFRC as a dictionary keyed by edge (u, v).
    """
    nodes = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    idx_to_node = {i: n for i, n in enumerate(nodes)}
    
    A = nx.adjacency_matrix(G, nodelist=nodes)
    F_star = compute_afrc(A, gamma)
    
    u, v = A.nonzero()
    
    afrc_dict = {}
    for i in range(len(u)):
        node_u = idx_to_node[u[i]]
        node_v = idx_to_node[v[i]]
        afrc_dict[(node_u, node_v)] = F_star[i]
        
    return afrc_dict
