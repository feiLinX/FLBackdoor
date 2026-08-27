import torch
import torch.nn.functional as F


def supcon_loss(features, labels, temperature=0.5):

    f = F.normalize(features, dim=1)
    sim = (f @ f.t()) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()        # numerical stability
    n = features.size(0)
    self_mask = torch.eye(n, device=features.device)
    labels = labels.contiguous().view(-1, 1)
    pos_mask = (labels == labels.t()).float() - self_mask       # same label, excl. self
    exp_sim = torch.exp(sim) * (1 - self_mask)                  # denominator excludes self
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_cnt = pos_mask.sum(dim=1)
    per_sample = -(pos_mask * log_prob).sum(dim=1) / pos_cnt.clamp(min=1)
    valid = pos_cnt > 0
    
    return per_sample[valid].mean() if valid.any() else features.sum() * 0.0
