import torch
import torch.nn.functional as F


def supcon_loss(features, labels, temperature=0.5):
    """Supervised contrastive loss used as a simplified Chameleon (Dai et al., ICML
    2023) 'adapt-to-peer-images' term. Pulls same-label feature vectors together and
    pushes different-label ones apart: the triggered samples (relabelled to the target)
    are thereby aligned with clean target-class images ('facilitators') and separated
    from images that share their true class ('interferers'), planting the backdoor in a
    feature cluster that later benign updates rarely overwrite (durability). Simplified
    -- we use only the in-batch supervised-contrastive term on the model's own features,
    not Chameleon's full two-stage contrastive-pretrain-then-poison schedule.

    features: [B, D] penultimate features; labels: [B] (triggered samples carry the
    target label). Returns a 0-connected tensor when a batch has no positive pairs.
    """
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
