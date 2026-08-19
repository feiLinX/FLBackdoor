import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Vanilla reuses the shared pattern-trigger backdoor builder (same 4-tuple as PGD /
# Neurotoxin); its only distinctive piece is the uncapped model replacement below.
from attacks.pgd import build_trigger_backdoor  # noqa: F401  (re-exported for convenience)


def apply_model_replacement(global_net, nets_current, adv_clients, scale=1.0):
    """Vanilla model-replacement backdoor (Bagdasaryan et al., "How to Backdoor
    Federated Learning", 2020). Boost each malicious client's update
    delta = w_local - w_global by `scale` with NO norm cap, so under plain FedAvg the
    scaled update overpowers the benign average (set scale ~ #clients to fully replace
    the global model). Unlike CDLS's constrain-and-scale it makes no attempt to stay
    norm-bounded, which is exactly why robust aggregators (Krum / NDC / FLAME / GRAID)
    filter it -- the naive baseline. Operates in place; only floating-point tensors are
    scaled (BN counters etc. are copied through unchanged)."""
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
