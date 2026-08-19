import os, sys
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aggregations import fedavg_global


def _graid_input_shape(args):

    if args.dataset in ('digits', 'cifar10', 'cifar100'):
        return 3, 32, 32
    return 3, 224, 224  # domain / office (expensive to invert -- see notes)


def _graid_num_classes(args):
    return 100 if args.dataset == 'cifar100' else 10


def _graid_input_bounds(args, device):

    if args.dataset == 'cifar10':
        mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
        std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
    elif args.dataset == 'cifar100':
        mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
        std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    elif args.dataset == 'digits':
        mean = [0.1307, 0.1307, 0.1307]
        std = [0.3081, 0.3081, 0.3081]
    else:  # domain / office -> ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    lo = (0.0 - mean) / std
    hi = (1.0 - mean) / std
    return lo, hi


def _total_variation(x):

    dh = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    dw = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return dh + dw


def _client_pseudo_gradient(global_net, w_local):

    vec = []
    for name, p in global_net.named_parameters():
        if p.requires_grad:
            vec.append((p.detach() - w_local[name].to(p.device, p.dtype)).reshape(-1))
    return torch.cat(vec)


def _split_suspicious(feats, min_samples, sep_ratio):

    n = feats.shape[0]
    if n < min_samples:
        return np.zeros(n, dtype=bool)

    # reduce dimension so the cluster geometry is stable with few samples
    d = int(min(feats.shape[1], n - 1, 3))
    X = PCA(n_components=d, random_state=0).fit_transform(feats) if feats.shape[1] > d else feats

    km = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X)
    lab = km.labels_
    if len(np.unique(lab)) < 2:
        return np.zeros(n, dtype=bool)

    # KMeans always returns 2 clusters, so accept the split only when the two are
    # well separated: (inter-centroid distance) / (mean intra-cluster radius) > sep_ratio.
    c0, c1 = km.cluster_centers_
    between = float(np.linalg.norm(c0 - c1))
    r0 = float(np.linalg.norm(X[lab == 0] - c0, axis=1).mean())
    r1 = float(np.linalg.norm(X[lab == 1] - c1, axis=1).mean())
    if between <= sep_ratio * (0.5 * (r0 + r1) + 1e-8):
        return np.zeros(n, dtype=bool)             # not separated enough -> homogeneous

    minority = int(np.argmin(np.bincount(lab, minlength=2)))
    return lab == minority


def _graid_filter(args, global_net, recon, client_ids, logger):

    device = next(global_net.parameters()).device

    imgs, labels, owners = [], [], []
    for cid in client_ids:
        dx, dy = recon[cid]
        hard = dy.argmax(dim=1)
        for k in range(dx.size(0)):
            imgs.append(dx[k])
            labels.append(int(hard[k]))
            owners.append(cid)

    imgs_t = torch.stack(imgs)
    n = imgs_t.size(0)
    flat = imgs_t.reshape(n, -1).cpu().numpy()
    labels = np.asarray(labels)
    owners = np.asarray(owners)
    suspicious = np.zeros(n, dtype=bool)

    # ---- step 4: within-class clustering to detect suspicious reconstructions
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        flag = _split_suspicious(flat[idx], args.def_min_cluster, args.def_sep_ratio)
        suspicious[idx[flag]] = True

    # ---- step 5: label-consistency filtering on the (currently) benign group.
    benign = np.where(~suspicious)[0]
    if len(benign) > 0:
        global_net.eval()
        with torch.no_grad():
            preds = global_net(imgs_t[benign].to(device)).argmax(dim=1).cpu().numpy()
        suspicious[benign[preds != labels[benign]]] = True

    # ---- step 6: discard clients whose suspicious fraction exceeds the threshold
    accepted = []
    for cid in client_ids:
        m = owners == cid
        frac = float(suspicious[m].mean()) if m.any() else 0.0
        if frac > args.def_susp_threshold:
            if logger is not None:
                logger.info('[GRAID] discard client %s (suspicious frac %.2f)' % (str(cid), frac))
        else:
            accepted.append(cid)

    if not accepted:
        if logger is not None:
            logger.info('[GRAID] all clients flagged this round -> keeping all (FedAvg fallback)')
        accepted = list(client_ids)

    return accepted


def reconstruct_client_batch(global_net, params, target_vec, dummy_x, dummy_y,
                             iters, lr, tv_weight, box=None):

    dummy_x = dummy_x.detach().clone().requires_grad_(True)
    dummy_y = dummy_y.detach().clone().requires_grad_(True)
    opt = optim.Adam([dummy_x, dummy_y], lr=lr)
    target_vec = target_vec.detach()

    for _ in range(iters):
        opt.zero_grad()
        out = global_net(dummy_x)                              # [B, num_classes]
        soft = F.softmax(dummy_y, dim=1)
        logp = F.log_softmax(out, dim=1)
        task_loss = -(soft * logp).sum(dim=1).mean()          # soft-label CE

        # dummy gradient wrt model params (create_graph -> differentiable wrt x, y)
        grads = torch.autograd.grad(task_loss, params, create_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        gvec = torch.cat([g.reshape(-1) for g in grads])

        rec_loss = 1.0 - F.cosine_similarity(gvec, target_vec, dim=0)
        loss = rec_loss + tv_weight * _total_variation(dummy_x)

        # grads wrt the dummy tensors only (do NOT pollute model .grad buffers)
        gx, gy = torch.autograd.grad(loss, [dummy_x, dummy_y])
        dummy_x.grad, dummy_y.grad = gx, gy
        opt.step()

        # box constraint (DLG): project the dummy image back onto the valid
        # pixel range (in normalized space) so it stays on the image manifold.
        if box is not None:
            with torch.no_grad():
                dummy_x.data = torch.maximum(torch.minimum(dummy_x, box[1]), box[0])

    return dummy_x.detach(), dummy_y.detach()


def graid_aggregate(args, global_net, nets_current, client2loaders, comm_round, logger):

    device = 'cuda'
    global_net = global_net.to(device)
    global_net.eval()
    params = [p for _, p in global_net.named_parameters() if p.requires_grad]

    C, H, W = _graid_input_shape(args)
    n_classes = _graid_num_classes(args)
    client_ids = list(nets_current.keys())

    # warm-up
    warmup = getattr(args, 'def_warmup', 0)
    if comm_round < warmup:
        if logger is not None:
            logger.info('[GRAID] round %d < warm-up %d: no screening -> FedAvg over all'
                        % (comm_round, warmup))
        global_net.to('cpu')
        fedavg_global(global_net, client2loaders, nets_current)
        return list(client_ids)

    # run the (expensive) reconstruction + screening only every def_recon_every
    # rounds; on the other rounds fall back to plain FedAvg over all clients.
    recon_every = max(1, getattr(args, 'def_recon_every', 1))
    if comm_round % recon_every != 0:
        if logger is not None:
            logger.info('[GRAID] round %d: skipping reconstruction (runs every %d rounds) -> FedAvg over all'
                        % (comm_round, recon_every))
        global_net.to('cpu')
        fedavg_global(global_net, client2loaders, nets_current)
        return list(client_ids)

    box = _graid_input_bounds(args, device)   # project dummy images onto the valid pixel range

    # ---- steps 1-3: per-client gradient-inversion reconstruction (fresh each round) ----
    recon = {}
    for cid in client_ids:
        w_local = nets_current[cid].state_dict()
        target_vec = _client_pseudo_gradient(global_net, w_local).detach()

        dummy_x = torch.randn(args.def_num_recon, C, H, W, device=device)
        dummy_y = torch.randn(args.def_num_recon, n_classes, device=device)
        dummy_x, dummy_y = reconstruct_client_batch(
            global_net, params, target_vec, dummy_x, dummy_y,
            iters=args.def_recon_iters, lr=args.def_recon_lr, tv_weight=args.def_tv_weight, box=box)
        recon[cid] = (dummy_x.detach(), dummy_y.detach())

    # ---- steps 4-6: screen and discard (runs every round; no warm-up) ----
    accepted = _graid_filter(args, global_net, recon, client_ids, logger)
    if logger is not None:
        logger.info('[GRAID] round %d accepted %d/%d clients: %s'
                    % (comm_round, len(accepted), len(client_ids), str(accepted)))

    # ---- aggregate over accepted clients (FedAvg) ----
    global_net.to('cpu')  # match fedavg_global, which mixes cpu client state_dicts
    accepted_nets = {cid: nets_current[cid] for cid in accepted}
    fedavg_global(global_net, client2loaders, accepted_nets)
    
    return accepted