import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj


class gnn_basic(nn.Module):
    def __init__(self, num_features_list):

        super(gnn_basic, self).__init__()
        num_layers = len(num_features_list)
        self.layers = nn.ModuleList()
        for i in range(num_layers-1):
            self.layers.append(TAGConv(num_features_list[i], num_features_list[i+1], K=5))

    def forward(self, x, edge_index, edge_weight):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index=edge_index, edge_weight=edge_weight)
            x = F.leaky_relu(x)

        return x


class GNN_Model(nn.Module):
    def __init__(self, num_feature_list):
        super(GNN_Model, self).__init__()
        self.gnn_basic = gnn_basic(num_feature_list)
        self.b_p = nn.Linear(num_feature_list[-1], 1, bias=False)
        self.w = nn.Linear(num_feature_list[-1], num_feature_list[-1], bias=False)
        # self.flows = K

    def forward(self, x, edge_index, edge_weight, K, batch_size, N, eigen, device):
        # r_init = []
        # a_init = []

        y = self.gnn_basic(x, edge_index, edge_weight)

        a_y = self.b_p(y)
        yw = self.w(y)
        yw = yw.view(K * batch_size, N, -1)
        y = y.view(K * batch_size, N, -1)

        r_y = torch.bmm(yw, torch.transpose(y, dim0=2, dim1=1))

        a0, mu_, path = torch.split(x, 1, dim=-1)

        a_ik = a0 + torch.relu(a_y)  ######## Third Constraint Implicit Satisfaction  ########
        a_ik = torch.squeeze(a_ik)
        a_ik = a_ik.view(K, -1)

        temp_edge_weight = edge_weight.view(K, batch_size*N*(N-1))
        temp_edge_weight = temp_edge_weight.view(K, batch_size, N*(N-1))
        temp_eigen = eigen.view(K, batch_size)

        C = torch.zeros(K, batch_size, N, N)
        for k in range(K):
            for b in range(batch_size):
                idx = edge_index[:,:N*(N-1)]
                wt = temp_edge_weight[k, b, :]
                batch = torch.zeros(N, dtype=torch.long, device=device)
                adj_matrix = to_dense_adj(idx, batch, wt, max_num_nodes=N)
                C1 = torch.squeeze(adj_matrix) * temp_eigen[k, b]
                C[k,b,:,:] = C1

        path = path.view(K*batch_size, -1)

        a_ik = a_ik.view(K*batch_size, -1)
        a_ik = a_ik * (1 - path)

        r_y[path == 1] = -1e10
        r_y = r_y.view(K, batch_size, N, N)
        r_k_softmax = torch.softmax(r_y, dim=0)

        C = C.to(device)
        r_ij = r_k_softmax * C

        r_ij = r_ij.view(K*batch_size, N, N)
        r_ji = torch.transpose(r_ij, dim0=2, dim1=1)

        a_ik = a_ik.view(K, batch_size, N)

        return r_ij, r_ji, a_ik


