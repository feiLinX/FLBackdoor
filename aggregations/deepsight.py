import copy
import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import HDBSCAN


def deepsight(global_net, client2loaders, nets_this_round,
              num_seeds=3, num_samples=2000, noise_bs=100, tau=1.0 / 3):

    device = 'cuda'
    client_ids = list(nets_this_round.keys())
    n = len(client_ids)

    global_net = global_net.to(device).eval()
    g_state = global_net.state_dict()

    # locate the output (last nn.Linear) layer -> its weight/bias keys + #classes
    last_name = None
    for name, module in global_net.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
    w_key, b_key = last_name + '.weight', last_name + '.bias'
    num_classes = g_state[w_key].shape[0]

    # input shape (C, H, W) read from a real client batch (dataset-agnostic)
    sample_x = next(iter(client2loaders[client_ids[0]]))[0]
    C, H, W = sample_x.shape[1], sample_x.shape[2], sample_x.shape[3]

    # ---- NEUPs (per-class output-update energy) and TEs (threshold exceedings) ----
    gw = g_state[w_key].detach().cpu().numpy()
    gb = g_state[b_key].detach().cpu().numpy()
    NEUPs, TEs = [], []
    for c in client_ids:
        st = nets_this_round[c].state_dict()
        diff_w = st[w_key].detach().cpu().numpy() - gw
        diff_b = st[b_key].detach().cpu().numpy() - gb
        UPs = np.abs(diff_b) + np.sum(np.abs(diff_w), axis=1)
        NEUP = UPs ** 2 / np.sum(UPs ** 2)
        TEs.append(int(np.sum(NEUP >= (1.0 / num_classes) * np.max(NEUP))))
        NEUPs.append(NEUP)
    NEUPs = np.array(NEUPs)
    # a LOW threshold-exceeding count marks a client as suspicious (label 1)
    labels = np.array([0 if te >= np.median(TEs) / 2 else 1 for te in TEs])

    # ---- DDifs: mean softmax ratio (local / global) over random-noise inputs ----
    temp_model = copy.deepcopy(global_net)
    DDifs = np.zeros((num_seeds, n, num_classes))
    for s in range(num_seeds):
        torch.manual_seed(s)
        for ci, c in enumerate(client_ids):
            temp_model.load_state_dict(nets_this_round[c].state_dict())
            temp_model.eval()
            DDif = torch.zeros(num_classes, device=device)
            seen = 0
            while seen < num_samples:
                b = min(noise_bs, num_samples - seen)
                x = torch.rand(b, C, H, W, device=device)
                with torch.no_grad():
                    out_l = torch.softmax(temp_model(x), dim=1)
                    out_g = torch.softmax(global_net(x), dim=1)
                DDif += torch.div(out_l, out_g + 1e-8).sum(dim=0)
                seen += b
            DDifs[s, ci] = (DDif / num_samples).cpu().numpy()
    del temp_model

    # ---- cosine distance between the clients' output-layer bias updates ----
    cos = nn.CosineSimilarity(dim=0, eps=1e-6)
    gb_t = g_state[b_key].detach().cpu()
    bias_updates = [nets_this_round[c].state_dict()[b_key].detach().cpu() - gb_t for c in client_ids]
    cosine_distance = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cosine_distance[i, j] = 1.0 - cos(bias_updates[i], bias_updates[j]).item()

    # ---- cluster each fingerprint, fuse the memberships, run a final clustering ----
    def _dists_from_clust(clusters):
        pd = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                pd[i, j] = 0.0 if clusters[i] == clusters[j] else 1.0
        return pd

    def _cluster(x):
        x = np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        return HDBSCAN(min_samples=1, allow_single_cluster=True).fit_predict(x)

    cosine_dists = _dists_from_clust(_cluster(cosine_distance))
    neup_dists = _dists_from_clust(_cluster(NEUPs))
    ddif_dists = np.mean([_dists_from_clust(_cluster(DDifs[s])) for s in range(num_seeds)], axis=0)
    merged = np.mean([ddif_dists, neup_dists, cosine_dists], axis=0)
    final_clusters = _cluster(merged)

    # ---- accept clients: keep low-suspicion clusters (+ non-suspicious noise pts) ----
    accepted_pos = []
    for cl in np.unique(final_clusters):
        idxs = np.argwhere(final_clusters == cl).flatten()
        if cl == -1:
            accepted_pos.extend(int(i) for i in idxs if labels[i] == 0)
        elif np.sum(labels[idxs]) / len(idxs) < tau:
            accepted_pos.extend(int(i) for i in idxs)

    if not accepted_pos:                        # safety valve: never drop everyone
        accepted_pos = list(range(n))
    accepted_ids = [client_ids[i] for i in accepted_pos]

    global_net.to('cpu')

    # ---- FedAvg over the accepted clients ----
    total = sum(len(client2loaders[c].dataset) for c in accepted_ids)
    freqs = [len(client2loaders[c].dataset) / total for c in accepted_ids]
    w_global = global_net.state_dict()
    for pos, c in enumerate(accepted_ids):
        net_para = nets_this_round[c].state_dict()
        if pos == 0:
            for key in net_para:
                w_global[key] = net_para[key] * freqs[pos]
        else:
            for key in net_para:
                w_global[key] += net_para[key] * freqs[pos]
    global_net.load_state_dict(w_global)
    return accepted_ids
