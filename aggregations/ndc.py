import torch


def ndc(global_net, client2loaders, nets_this_round, norm_threshold=None):

    client_ids = list(nets_this_round.keys())
    nets = list(nets_this_round.values())
    n = len(nets)

    # flatten trainable params -> per-client update norm ||w_i - w_global||
    g_vec = torch.cat([p.detach().reshape(-1) for p in global_net.parameters()])
    vecs = [torch.cat([p.detach().reshape(-1) for p in net.parameters()]) for net in nets]
    update_norms = [torch.norm(v - g_vec) for v in vecs]

    # clip bound M: fixed if provided, else the median update norm (adaptive)
    if norm_threshold is None:
        M = torch.stack(update_norms).median()
    else:
        M = torch.as_tensor(float(norm_threshold))

    # per-client scaling factor gamma_i = min(1, M / ||delta_i||)
    gammas = [min(1.0, (M / (update_norms[i] + 1e-12)).item()) for i in range(n)]

    # weighted (fedavg-style) average of the clipped updates
    total_data_points = sum(len(client2loaders[r].dataset) for r in client_ids)
    freqs = [len(client2loaders[client_ids[i]].dataset) / total_data_points for i in range(n)]

    g_state = global_net.state_dict()
    w_global = {k: v.clone().float() for k, v in g_state.items()}
    for i in range(n):
        net_para = nets[i].state_dict()
        for key in w_global:
            w_global[key] += freqs[i] * gammas[i] * (net_para[key].float() - g_state[key].float())

    global_net.load_state_dict({k: w_global[k].to(g_state[k].dtype) for k in g_state})