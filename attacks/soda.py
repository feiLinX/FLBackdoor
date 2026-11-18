import os
import sys
import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import get_dataloader
from attacks.cdls import InMemoryImageDataset, get_cdls_transforms, _load_cdls_raw_images, _pil_resize_like


def soda_self_reference(global_net, clean_loader, args):
    """SoDa self-reference stage: train a copy of the global model on a malicious client's clean data."""
    ref = copy.deepcopy(global_net).cuda()
    ref.train()
    opt = optim.SGD(ref.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    criterion = nn.CrossEntropyLoss()
    for _ in range(args.epochs):
        for x, target in clean_loader:
            x, target = x.cuda(), target.cuda()
            opt.zero_grad()
            criterion(ref(x), target).backward()
            opt.step()
    return torch.nn.utils.parameters_to_vector([p.detach() for p in ref.parameters()]).detach()


def build_soda_backdoor(args, client2dataidx, adv_clients, target_label, poison_frac,
                        dataset='digits', domain=None, ood_dataset='cifar10', ood_domain=None,
                        ood_test_size=2000, seed=0):

    if dataset == 'digits' and domain is None:
        domain = args.bd_domain

    transform_train, transform_test = get_cdls_transforms(args, dataset)
    rng = random.Random(seed)

    # OOD image pools (raw HWC uint8); resized to the victim's spatial size on the fly
    ood_train_imgs, _ = _load_cdls_raw_images(args, ood_dataset, ood_domain, None, train=True)
    ood_test_imgs, _ = _load_cdls_raw_images(args, ood_dataset, ood_domain, None, train=False)

    client2loaders, client2clean_loaders = {}, {}
    train_poison_images, train_poison_labels = [], []
    for client_id in range(args.nclients):
        dataidxs = client2dataidx[client_id]
        if client_id in adv_clients:
            images, labels = _load_cdls_raw_images(args, dataset, domain, dataidxs, train=True)

            # clean loader for the self-reference (built BEFORE poisoning)
            clean_ds = InMemoryImageDataset([im.copy() for im in images], list(labels), transform=transform_train)
            client2clean_loaders[client_id] = DataLoader(clean_ds, batch_size=args.batch_size,
                                                         drop_last=True, shuffle=True, num_workers=4)


            n_poison = int(round(poison_frac * len(images)))
            victim_idx = rng.sample(range(len(images)), n_poison) if n_poison > 0 else []
            for i in victim_idx:
                ood = ood_train_imgs[rng.randrange(len(ood_train_imgs))]
                images[i] = _pil_resize_like(ood, images[i].shape[:2])
                labels[i] = int(target_label)
                train_poison_images.append(images[i])
                train_poison_labels.append(int(target_label))

            train_ds = InMemoryImageDataset(images, labels, transform=transform_train)
            train_dl = DataLoader(train_ds, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=4)
        else:
            train_dl, _ = get_dataloader(args, dataset=dataset, data_dir=args.data_dir,
                                          train_bs=args.batch_size, test_bs=args.batch_size, dataidxs=dataidxs)
        client2loaders[client_id] = train_dl

    train_poison_dl = DataLoader(
        InMemoryImageDataset(train_poison_images, train_poison_labels, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4) if train_poison_images else None


    test_images, test_labels = _load_cdls_raw_images(args, dataset, domain, None, train=False)
    clean_test_dl = DataLoader(
        InMemoryImageDataset([im.copy() for im in test_images], list(test_labels), transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    tgt_hw = test_images[0].shape[:2]
    n_bd = min(len(ood_test_imgs), ood_test_size)
    bd_sel = rng.sample(range(len(ood_test_imgs)), n_bd)
    bd_images = [_pil_resize_like(ood_test_imgs[i], tgt_hw) for i in bd_sel]
    bd_labels = [int(target_label)] * n_bd
    backdoor_test_dl = DataLoader(
        InMemoryImageDataset(bd_images, bd_labels, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    return client2loaders, clean_test_dl, backdoor_test_dl, train_poison_dl, client2clean_loaders
