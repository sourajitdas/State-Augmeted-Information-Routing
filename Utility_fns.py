import torch


def source_destination(N):
    length = N-1
    final = torch.zeros(length)
    index = torch.randint(0, length, (1,))
    final[index] = 1
    return final


def flow_path(N, K):
    temp_path = torch.zeros(K, N-1)
    for k in range(K):
        # print('k', k)
        final = source_destination(N)
        while torch.sum(torch.all(temp_path[:k, :] == final, dim=1)):
            final = source_destination(N)
        temp_path[k, :] = final

    final_path = torch.transpose(temp_path, dim0=0, dim1=1)
    x = torch.zeros(1,K)

    ultimate_path = torch.cat((x, final_path), dim=0)

    return ultimate_path
