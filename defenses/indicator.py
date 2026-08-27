import os, sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aggregations import fedavg_global
from attacks.cdls import InMemoryImageDataset, get_cdls_transforms, _load_cdls_raw_images, _pil_resize_like


def build_indicator_data(args, ood_dataset=None, ood_domain=None, n_samples=200, seed=0):

    if ood_dataset is None:
        ood_dataset = 'digits' if args.dataset == 'cifar10' else 'cifar10'
    if ood_domain is None:
        ood_domain = 'mnist' if ood_dataset == 'digits' else None

    ood_imgs, _ = _load_cdls_raw_images(args, ood_dataset, ood_domain, None, train=True)
    rng = random.Random(seed)
    sel = rng.sample(range(len(ood_imgs)), min(n_samples, len(ood_imgs)))

    ind_hw = (32, 32) if args.dataset in ('digits', 'cifar10', 'cifar100') else (224, 224)
    num_classes = 100 if args.dataset == 'cifar100' else 10

    imgs = [_pil_resize_like(ood_imgs[i], ind_hw) for i in sel]
    labels = [rng.randrange(num_classes) for _ in sel]

    _, transform_test = get_cdls_transforms(args, args.dataset)
    ds = InMemoryImageDataset(imgs, labels, transform=transform_test)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4)


def inject_indicator(global_net, indicator_loader, lr=0.01, epochs=10, momentum=0.9, wd=5e-4, mu=0.0):

    device = 'cuda'
    global_net = global_net.to(device)

    orig_bn = {k: v.detach().clone() for k, v in global_net.state_dict().items() if 'running_' in k}
    ref = {n: p.detach().clone() for n, p in global_net.named_parameters()} if mu > 0 else None

    global_net.train()
    optimizer = optim.SGD(global_net.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in indicator_loader:
            x, y = x.to(device), y.to(dtype=torch.int64, device=device)
            optimizer.zero_grad()
            loss = criterion(global_net(x), y)
            if mu > 0:
                loss = loss + (mu / 2) * sum(((p - ref[n]) ** 2).sum() for n, p in global_net.named_parameters())
            loss.backward()
            optimizer.step()

    # post-injection BN stats carry the watermark -> hand them to indicator()
    wm_bn = {k: v.detach().clone() for k, v in global_net.state_dict().items() if 'running_' in k}
    sd = global_net.state_dict()
    for k in orig_bn:
        sd[k].copy_(orig_bn[k])

    global_net.to('cpu')
    return {k: v.cpu() for k, v in wm_bn.items()}


def indicator(global_net, client2loaders, nets_this_round, indicator_loader, wm_bn=None,
              threshold=0.5, logger=None):

    device = 'cuda'
    client_ids = list(nets_this_round.keys())

    check_model = copy.deepcopy(global_net).to(device)
    check_model.eval()

    accepted = []
    for cid in client_ids:
        client_sd = nets_this_round[cid].state_dict()
        check_sd = check_model.state_dict()
        for k in check_sd:
            src = wm_bn[k] if (wm_bn is not None and k in wm_bn) else client_sd[k]
            check_sd[k].copy_(src.to(check_sd[k].device))

        correct, total = 0, 0
        with torch.no_grad():
            for x, y in indicator_loader:
                x, y = x.to(device), y.to(device)
                pred = check_model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        wm_acc = correct / max(total, 1)

        if wm_acc >= threshold:
            if logger is not None:
                logger.info('[Indicator] discard client %s (watermark preserved, acc %.3f)' % (str(cid), wm_acc))
        else:
            accepted.append(cid)
            if logger is not None:
                logger.info('[Indicator] keep client %s (watermark erased, acc %.3f)' % (str(cid), wm_acc))

    del check_model

    if not accepted:  # safety valve: never drop everyone
        if logger is not None:
            logger.info('[Indicator] all clients flagged this round -> keeping all (FedAvg fallback)')
        accepted = list(client_ids)

    accepted_nets = {cid: nets_this_round[cid] for cid in accepted}
    fedavg_global(global_net, client2loaders, accepted_nets)
    return accepted
