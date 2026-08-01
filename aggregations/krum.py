import torch


def krum(global_net, client2loaders, nets_this_round, nbyz=0, m=1):

    client_ids = list(nets_this_round.keys())
    nets = list(nets_this_round.values())
    n = len(nets)

    vectors = [torch.cat([p.detach().reshape(-1) for p in net.parameters()]) for net in nets]

    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.sum((vectors[i] - vectors[j]) ** 2)
            dists[i, j] = d
            dists[j, i] = d

    k = max(n - nbyz - 2, 1)
    scores = torch.empty(n)
    for i in range(n):
        neighbours = torch.sort(dists[i])[0][1:k + 1]  # [0] is the self-distance
        scores[i] = neighbours.sum()

    # how many clients survive: m=1 is Krum, m>1 is Multi-Krum, m=None keeps n-nbyz
    n_keep = max(n - nbyz, 1) if m is None else m
    selected_idx = torch.argsort(scores)[:n_keep].tolist()
    selected_ids = [client_ids[i] for i in selected_idx]

    # ---- from here it's exactly fedavg_global, but only over the survivors ----
    total_data_points = sum([len(client2loaders[r].dataset) for r in selected_ids])
    fed_avg_freqs = [len(client2loaders[r].dataset) / total_data_points for r in selected_ids]

    w_global = global_net.state_dict()
    for net_id, r in enumerate(selected_ids):
        net_para = nets_this_round[r].state_dict()
        if net_id == 0:
            for key in net_para:
                w_global[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                w_global[key] += net_para[key] * fed_avg_freqs[net_id]
                
    global_net.load_state_dict(w_global)
