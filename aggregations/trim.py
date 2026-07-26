import torch


def trimmed_mean(global_net, nets_this_round, b=0):

    nets = list(nets_this_round.values())
    n = len(nets)
    b = min(b, (n - 1) // 2)  

    w_global = global_net.state_dict()
    state_dicts = [net.state_dict() for net in nets]
    for key in w_global:
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        sorted_vals, _ = torch.sort(stacked, dim=0)   # ascending per coordinate
        trimmed = sorted_vals[b:n - b]                # drop b smallest & b largest
        w_global[key] = trimmed.mean(dim=0).to(w_global[key].dtype)
        
    global_net.load_state_dict(w_global)