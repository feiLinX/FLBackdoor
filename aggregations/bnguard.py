import torch
import torch.nn as nn
import numpy as np


def bnguard(global_net, client2loaders, nets_this_round):

    from sklearn.cluster import KMeans

    client_ids = list(nets_this_round.keys())
    n = len(client_ids)

    features = []
    for c in client_ids:
        first_bn = None
        for _, module in nets_this_round[c].named_modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                first_bn = module
                break
        rm, rv = first_bn.running_mean, first_bn.running_var
        features.append([torch.mean(rv).item(), torch.var(rv).item(),
                         torch.mean(rm).item(), torch.var(rm).item()])
    features = np.array(features)

    labels = KMeans(n_clusters=2, n_init=10).fit(features).labels_
    benign_class = int(np.argmax([np.sum(labels == 0), np.sum(labels == 1)]))
    benign_ids = [client_ids[i] for i in range(n) if labels[i] == benign_class]

    # FedAvg over the benign cluster
    total = sum(len(client2loaders[c].dataset) for c in benign_ids)
    freqs = [len(client2loaders[c].dataset) / total for c in benign_ids]
    w_global = global_net.state_dict()
    for pos, c in enumerate(benign_ids):
        net_para = nets_this_round[c].state_dict()
        if pos == 0:
            for key in net_para:
                w_global[key] = net_para[key] * freqs[pos]
        else:
            for key in net_para:
                w_global[key] += net_para[key] * freqs[pos]
    global_net.load_state_dict(w_global)
    return benign_ids
