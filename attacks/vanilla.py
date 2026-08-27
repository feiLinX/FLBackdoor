import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.pgd import build_trigger_backdoor  


def apply_model_replacement(global_net, nets_current, adv_clients, scale=1.0):

    adv_clients = set(adv_clients)
    g = global_net.state_dict()
    for cid in nets_current:
        if cid not in adv_clients:
            continue
        sd = nets_current[cid].state_dict()
        new_sd = {}
        for k in sd:
            if torch.is_floating_point(sd[k]):
                new_sd[k] = (g[k].float() + scale * (sd[k].float() - g[k].float())).to(sd[k].dtype)
            else:
                new_sd[k] = sd[k]
        nets_current[cid].load_state_dict(new_sd)
