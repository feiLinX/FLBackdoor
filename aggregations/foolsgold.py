import torch
import numpy as np

def foolsgold(global_net, client2loaders, nets_this_round):

    from sklearn.metrics.pairwise import cosine_similarity

    client_ids = list(nets_this_round.keys())
    nets = [nets_this_round[c] for c in client_ids]
    n = len(nets)

    g_vec = torch.cat([p.detach().reshape(-1) for p in global_net.parameters()])
    updates = [torch.cat([p.detach().reshape(-1) for p in net.parameters()]) - g_vec for net in nets]
    grads = torch.stack(updates).cpu().numpy()

    cs = cosine_similarity(grads) - np.eye(n)
    maxcs = np.max(cs, axis=1)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]

    wv = 1 - np.max(cs, axis=1)
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    wv = wv / np.max(wv)                 # rescale so the most-trusted client has weight 1
    wv[wv == 1] = 0.99
    wv = np.log(wv / (1 - wv)) + 0.5     # logit re-weighting
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[wv < 0] = 0

    g_state = global_net.state_dict()
    w_global = {k: v.clone().float() for k, v in g_state.items()}
    for i in range(n):
        net_para = nets[i].state_dict()
        for key in w_global:
            w_global[key] += (wv[i] / n) * (net_para[key].float() - g_state[key].float())

    global_net.load_state_dict({k: w_global[k].to(g_state[k].dtype) for k in g_state})
