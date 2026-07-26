import torch
import numpy as np
from sklearn.cluster import HDBSCAN


def flame(global_net, client2loaders, nets_this_round, lamda=0.001, add_noise=True):

    client_ids = list(nets_this_round.keys())
    nets = list(nets_this_round.values())
    n = len(nets)

    # flatten trainable params -> geometry for clustering / clipping
    g_vec = torch.cat([p.detach().reshape(-1) for p in global_net.parameters()])
    vecs = [torch.cat([p.detach().reshape(-1) for p in net.parameters()]) for net in nets]
    update_norms = [torch.norm(v - g_vec) for v in vecs]

    normed = torch.nn.functional.normalize(torch.stack(vecs), dim=1)
    cos_dist = (1 - normed @ normed.t()).cpu().numpy().astype(np.float64)
    np.clip(cos_dist, 0, None, out=cos_dist)  # kill tiny negative float errors
    np.fill_diagonal(cos_dist, 0.0)           
    labels = HDBSCAN(min_cluster_size=n // 2 + 1, min_samples=1,
                     allow_single_cluster=True, metric='precomputed').fit_predict(cos_dist)
    if labels.max() < 0:
        benign = list(range(n))  # no cluster found -> accept everyone
    else:
        major = np.bincount(labels[labels >= 0]).argmax()
        benign = [i for i in range(n) if labels[i] == major]

    clip_value = torch.stack([update_norms[i] for i in benign]).median()
    gammas = [min(1.0, (clip_value / update_norms[i]).item()) for i in benign]

    total = sum(len(client2loaders[client_ids[i]].dataset) for i in benign)
    freqs = [len(client2loaders[client_ids[i]].dataset) / total for i in benign]

    g_state = global_net.state_dict()
    w_global = {k: v.clone().float() for k, v in g_state.items()}
    for freq, gamma, i in zip(freqs, gammas, benign):
        net_para = nets[i].state_dict()
        for key in w_global:
            w_global[key] += freq * gamma * (net_para[key].float() - g_state[key].float())

    if add_noise:
        std = (lamda * clip_value).item()
        for key in w_global:
            if not any(s in key for s in ('running_mean', 'running_var', 'num_batches_tracked')):
                w_global[key] += torch.randn_like(w_global[key]) * std

    global_net.load_state_dict(w_global)
