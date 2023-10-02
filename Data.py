import numpy as np
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from collections import defaultdict
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy import sparse
from Channel import create_network
from Utility_fns import flow_path


def create_graph_data(samples, N, K, T, batch_size):
    data_list = defaultdict(list)
    loader = {}

    for phase in samples:
        for _ in range(samples[phase]):
            all_A0s = []
            all_path = []
            all_edge_indices = []
            all_edge_weights = []
            all_eigenvalues = []

            graph_mat = create_network(N)
            np.fill_diagonal(graph_mat, 0)
            eigenvalues_t, _ = np.linalg.eig(graph_mat)
            eigen_max = np.max(eigenvalues_t)
            graph = graph_mat / eigen_max

            for t in range(T):
                A0_t = torch.rand(N,K)
                all_A0s.append(A0_t)

                path = flow_path(N, K)
                all_path.append(path)

                edge_index_t, edge_weights_t = from_scipy_sparse_matrix(sparse.csr_matrix(graph))

                all_edge_indices.append(edge_index_t)
                all_edge_weights.append(edge_weights_t)
                all_eigenvalues.append(eigen_max)

            X = torch.stack(all_A0s, dim=2)
            path = torch.stack(all_path, dim=2)

            GSO = Data(x=X, edge_index=all_edge_indices, edge_attr=all_edge_weights, eig=all_eigenvalues, map=path)

            data_list[phase].append(GSO)

        loader[phase] = DataLoader(data_list[phase], batch_size=batch_size, shuffle=(phase == 'train'))

    return loader

