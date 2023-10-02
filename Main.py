import numpy as np
import torch
import random
import torch.cuda
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from Data import create_graph_data
from SA_GNN_model import GNN_Model

random_seed = 1357537
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)


def train_eval(epochs, model, batch_size, dual_step, xi):
    pbar = tqdm(range(epochs), desc=f"Training for n = {N} nodes and k = {K} flows")

    loader = create_graph_data(num_samples, N, K, T,  batch_size)
    all_epoch_results = defaultdict(list)
    for epoch in pbar:
        print('epoch= ', epoch)
        for phase in loader:
            if phase == 'train':
                print("\nEntered Training Phase")
                model.train()
            else:
                print("\nEntered Evaluation Phase")
                model.eval()

            all_variables = defaultdict(list)
            for data in loader[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):

                    all_rij = []
                    all_rji = []
                    all_aik = []

                    avg_rij = []
                    avg_rji = []
                    avg_aik = []
                    avg_info = []
                    avg_quelen = []

                    all_info = []
                    all_queue = []

                    q_ik = torch.zeros(K*batch_size*N, 1, device=device_arg)

                    if phase == 'train':
                        mu_dual = torch.rand(K*batch_size*N, 1, device=device_arg)
                    else:
                        mu_dual = torch.zeros(K*batch_size*N, 1, device=device_arg)

                    test_mu_t = []
                    phi_t = []
                    queue_t = []
                    a_t = []
                    store_aik = []
                    for t in range(T):
                        data = data.to(device_arg)
                        A0_t = data.x[:,:,t].T
                        A0_t = A0_t.reshape(-1,1)

                        path = data.map[:, :, t].T
                        path = path.reshape(-1, 1)

                        tensor_eig = torch.tensor(data.eig)
                        single_eig = tensor_eig[:,t]
                        single_eig = single_eig.reshape(-1,1)

                        single_flow_edge_index = data.edge_index[t].to(torch.int64)
                        single_flow_edge_weight = data.edge_attr[t].to(torch.float)

                        edge_index = []
                        edge_weight = []
                        eigen = []
                        for flow in range(K):

                            flow_edge_index = single_flow_edge_index + (flow * batch_size * N)
                            edge_index.append(flow_edge_index)

                            flow_eigen = single_eig
                            eigen.append(flow_eigen)

                            flow_edge_weight = single_flow_edge_weight
                            edge_weight.append(flow_edge_weight)

                        # print('After concatenating')
                        edge_index = torch.cat(edge_index, dim=1)
                        edge_weight = torch.cat(edge_weight, dim=0)
                        eigen = torch.cat(eigen, dim=0)

                        path = path.view(K * batch_size * N, -1)

                        A0_t = A0_t * (1 - path)
                        x = torch.cat((A0_t, mu_dual, path), dim=1)

                        r_ij, r_ji, a_ik = model(x, edge_index, edge_weight, K, batch_size, N, eigen, device_arg)

                        phi = torch.sum((torch.sum(torch.log(a_ik + 10e-10), dim=0)), dim=-1)

                        temp_aik = a_ik.view(K * batch_size, N)
                        q_ik.data = torch.relu(q_ik.view(K * batch_size * N, -1) + A0_t -  \
                                               torch.sum(r_ij, dim=-1).view(K * batch_size * N, -1) + \
                                               torch.sum(r_ji, dim=-1).view(K * batch_size * N, -1))

                        q_ik = torch.squeeze(q_ik)
                        q_ik = q_ik.view(K, batch_size, N)
                        q_len = torch.sum(torch.sum(q_ik, dim=0), dim=-1)

                        all_rij.append(r_ij)
                        all_rji.append(r_ji)
                        all_aik.append(a_ik)
                        all_info.append(phi)
                        all_queue.append(q_len)
                        phi_t.append(phi.detach().cpu())
                        queue_t.append(q_ik.view(K * batch_size * N).detach().cpu())
                        a_t.append(a_ik.view(K * batch_size * N).detach().cpu())

                        if phase != 'train':
                            if (t+1) % T0 == 0:
                                test_rij = torch.stack(all_rij[-T0:], dim=0)
                                test_rji = torch.stack(all_rji[-T0:], dim=0)
                                test_aik = torch.stack(all_aik[-T0:], dim=0)

                                recent_rij = torch.mean(test_rij, dim=0)
                                recent_rji = torch.mean(test_rji, dim=0)
                                recent_aik = torch.mean(test_aik, dim=0)

                                sum_rij = torch.sum(recent_rij, dim=-1)
                                sum_rji = torch.sum(recent_rji, dim=-1)

                                recent_aik = recent_aik.view(K*batch_size, -1)

                                sum_rij = sum_rij.view(K * batch_size * N, -1)

                                sum_rji = sum_rji.view(K * batch_size * N, -1)
                                recent_aik = recent_aik.view(K*batch_size*N, -1)
                                xi = xi.view(K * batch_size * N, -1)

                                mu_dual.data = torch.relu(mu_dual - dual_step*(sum_rij - sum_rji - recent_aik - xi))

                                test_mu_t.append(mu_dual.detach().cpu())
                                store_aik.append(recent_aik.view(K,batch_size,N))

                    if phase != 'train':
                        test_mu_t = torch.stack(test_mu_t, dim=0)
                        queue_t = torch.stack(queue_t, dim=0)
                        phi_t = torch.stack(phi_t, dim=0)
                        a_t = torch.stack(a_t, dim=0)

                    all_rij = torch.stack(all_rij, dim=0)
                    all_rji = torch.stack(all_rji, dim=0)
                    all_aik = torch.stack(all_aik, dim=0)
                    all_info = torch.stack(all_info, dim=0)
                    all_queue = torch.stack(all_queue, dim=0)

                    avg_rij = torch.mean(all_rij, dim=0)
                    avg_rji = torch.mean(all_rji, dim=0)
                    avg_aik = torch.mean(all_aik, dim=0)

                    avg_info = torch.mean(all_info, dim=0)
                    avg_quelen = torch.mean(all_queue, dim=0)

                    if phase == 'train':
                        mu_dual = torch.squeeze(mu_dual)
                        mu_dual = mu_dual.view(K, batch_size * N)
                        mu_dual = mu_dual.view(K, batch_size, N)
                        xi = xi.view(K, batch_size, N)

                        avg_rij = avg_rij.view(K, batch_size, N, N)
                        avg_rji = avg_rji.view(K, batch_size, N, N)
                        avg_aik = avg_aik.view(K, batch_size, N)

                        U = torch.sum((torch.sum(torch.log(avg_aik + 10e-10), dim=0)), dim=-1)

                        T2 = mu_dual * (torch.sum(avg_rij, dim=-1) - torch.sum(avg_rji, dim=-1) - avg_aik - xi)
                        T3 = torch.sum(avg_rij, dim=-1) - torch.sum(avg_rji, dim=-1) - avg_aik - xi

                        L = -(U + torch.sum(torch.sum(T2 - 0.5 * dual_step * torch.square(T3), dim=0), dim=-1)).mean()

                        L.backward()
                        optimizer.step()

                all_variables['info'].extend(avg_info.detach().cpu().numpy().tolist())
                all_variables['queue'].extend(avg_quelen.detach().cpu().numpy().tolist())
                all_variables['queue_along_time'] = torch.mean(all_queue, dim=1).detach().cpu().numpy().tolist()
                all_variables['phi_along_time'] = torch.mean(all_info, dim=1).detach().cpu().numpy().tolist()

                if phase != 'train':

                    all_variables['mu_over_time'].append(test_mu_t.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['test_mu_over_time'].append(torch.mean(test_mu_t.squeeze(-1).T, dim=0).detach().cpu().numpy())

                    all_variables['average_queue_over_time'].append(torch.mean(queue_t.squeeze(-1).T, dim=0).detach().cpu().numpy())
                    all_variables['queue_over_time'].append(queue_t.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['phi_over_time'].append(phi_t.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['a_over_time'].append(a_t.squeeze(-1).T.detach().cpu().numpy())

                    all_variables['all_info'].append(all_info.squeeze(-1).T.detach().cpu().numpy())
                    all_variables['all_queue_len'].append(all_queue.squeeze(-1).T.detach().cpu().numpy())

            scheduler.step(L)

            for key in all_variables:
                if key == 'info':
                    all_epoch_results[phase, 'info_mean'].append(np.mean(all_variables['info']))
                    all_epoch_results[phase, 'info_max'].append(np.max(all_variables['info']))

                elif key == 'queue':
                    all_epoch_results[phase, 'queue_mean'].append(np.mean(all_variables['queue']))

                elif key in ['test_mu_over_time', 'mu_over_time']:
                    all_epoch_results[phase, 'test_mu_over_time'] = all_variables['test_mu_over_time']
                    all_epoch_results[phase, 'mu_over_time'] = all_variables['mu_over_time']

                elif key in ['queue_over_time']:
                    all_epoch_results[phase, 'queue_over_time'] = all_variables['queue_over_time']

                elif key in ['phi_over_time']:
                    all_epoch_results[phase, 'phi_over_time'] = all_variables['phi_over_time']

                elif key in ['a_over_time']:
                    all_epoch_results[phase, 'a_over_time'] = all_variables['a_over_time']

                elif key in ['average_queue_over_time']:
                    all_epoch_results[phase, 'average_queue_over_time'] = all_variables['average_queue_over_time']

                elif key in ['phi_along_time', 'queue_along_time']:
                    all_epoch_results[phase, 'phi_along_time'] = all_variables['phi_along_time']
                    all_epoch_results[phase, 'queue_along_time'] = all_variables['queue_along_time']

                else:
                    all_epoch_results[phase, key].append(np.mean(all_variables[key]))

    return all_epoch_results

N = 10
T = 100
T0 = 5
K = 5
device_arg = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_Size = 16
nEpochs = 40
nTrain = 128
nTest = 16
num_samples = {'train': nTrain, 'eval': nTest}
last_feature = 8
num_features_list = [3] + [32] + [last_feature]


xi_aux = torch.rand(K * batch_Size * N, 1, device=device_arg)
hk_primal_step = 0.005
mu_dual_step = 0.005

gnn_model = GNN_Model(num_features_list).to(device_arg)

optimizer = torch.optim.Adam(list(gnn_model.parameters()) + [xi_aux], hk_primal_step)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1)

results = train_eval(nEpochs, gnn_model, batch_Size, mu_dual_step, xi_aux)

mean_info = results['train', 'info_mean']
mean_info_eval = results['eval', 'info_mean']

max_info_eval = results['eval', 'info_max']

mean_queue_train = results['train', 'queue_mean']
mean_queue_eval = results['eval', 'queue_mean']
queue_ind = results['eval', 'queue_over_time']
phi_ind = results['eval', 'phi_over_time']
mu_ind = results['eval', 'mu_over_time']
a_ind = results['eval', 'a_over_time']

mu_plot = results['eval', 'test_mu_over_time']
queue_plot = results['eval', 'average_queue_over_time']

info_along = results['eval', 'phi_along_time']
queue_along = results['eval', 'queue_along_time']

plt.figure(0)
plt.subplot(2,1,1)
plt.plot(mean_info_eval, label='Evaluation Utility')
plt.title("GNN+MoM: Evaluation Utility for N=%d, primal step=%0.4f" % (N, hk_primal_step))
plt.xlabel("Epochs")
plt.ylabel("Mean value")
plt.legend()
plt.grid()
plt.tight_layout()
plt.subplot(2,1,2)
plt.plot(mean_queue_eval, label='Evaluation Queue length')
plt.title("Queue length for K=%d, dual step=%0.4f" % (K, mu_dual_step))
plt.xlabel("Epochs")
plt.ylabel("Max value")
plt.legend()
plt.grid()
plt.tight_layout()

plt.figure(1)
plt.subplot(2,1,1)
for i in range(K*nTest*N):
    plt.plot(mu_ind[0][i])
plt.title("Dual variables, mu for T=%d, N=%d, K=%d" % (T, N, K))
plt.xlabel("T/T0")
plt.ylabel("Max value")
plt.grid()
plt.tight_layout()
plt.subplot(2,1,2)
for i in range(K*nTest*N):
    plt.plot(queue_ind[0][i])
plt.title("Queue lengths for T0=%d, primal step=%0.4f and dual step=%0.4f" % (T0,hk_primal_step, mu_dual_step))
plt.xlabel("T/T0")
plt.ylabel("Max value")
plt.grid()
plt.tight_layout()

plt.show()



