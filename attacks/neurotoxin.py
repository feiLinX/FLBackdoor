import torch
import torch.nn as nn

def compute_neurotoxin_mask(global_net, clean_loader, mask_ratio, n_batches=5):

    device = 'cuda'
    global_net = global_net.to(device)
    was_training = global_net.training
    global_net.eval()               # use running BN stats; don't corrupt them
    global_net.zero_grad()
    criterion = nn.CrossEntropyLoss()

    seen = 0
    for x, target in clean_loader:
        x, target = x.to(device), target.to(dtype=torch.int64).to(device)
        loss = criterion(global_net(x), target)
        loss.backward()
        seen += 1
        if seen >= n_batches:
            break

    grads = [(p.grad.detach().abs() if p.grad is not None else torch.zeros_like(p)) for p in global_net.parameters()]
    flat = torch.cat([g.reshape(-1) for g in grads])
    k = int(mask_ratio * flat.numel())
    if k > 0:
        thresh = torch.topk(flat, k, largest=True).values.min()
        masks = [(g < thresh).float() for g in grads]      # 0 on the top-k benign coords, 1 elsewhere
    else:
        masks = [torch.ones_like(g) for g in grads]

    global_net.zero_grad()
    if was_training:
        global_net.train()
    global_net.to('cpu')
    
    return [m.detach().cpu() for m in masks]