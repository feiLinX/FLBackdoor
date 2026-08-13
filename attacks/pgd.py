import random
import os
import sys
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import get_dataloader
from attacks.cdls import InMemoryImageDataset, get_cdls_transforms, _load_cdls_raw_images


def add_trigger(img, trigger_size=5, trigger_value=255):

    img = img.copy()
    img[-trigger_size:, -trigger_size:, :] = trigger_value

    return img


def poison_trigger(images, labels, target_label, poison_frac, trigger_size, trigger_value, seed):

    rng = random.Random(seed)
    cand = [i for i in range(len(labels)) if int(labels[i]) != int(target_label)]
    n_poison = int(round(len(cand) * poison_frac))
    victim_idx = rng.sample(cand, n_poison) if n_poison > 0 else []
    for i in victim_idx:
        images[i] = add_trigger(images[i], trigger_size, trigger_value)
        labels[i] = int(target_label)

    return victim_idx


def build_trigger_backdoor(args, client2dataidx, adv_clients, target_label, poison_frac,
                           dataset='digits', domain=None, trigger_size=5, trigger_value=255, seed=0):

    if dataset == 'digits' and domain is None:
        domain = args.bd_domain
    elif dataset == 'domain' and domain is None:
        domain = 'clipart'

    transform_train, transform_test = get_cdls_transforms(args, dataset)

    # ---- train side: poison the malicious clients' own data ----
    client2loaders = {}
    train_poison_images, train_poison_labels = [], []
    for client_id in range(args.nclients):
        dataidxs = client2dataidx[client_id]
        if client_id in adv_clients:
            images, labels = _load_cdls_raw_images(args, dataset, domain, dataidxs, train=True)
            poisoned_idx = poison_trigger(images, labels, target_label, poison_frac,
                                          trigger_size, trigger_value, seed + client_id)
            train_poison_images.extend(images[i] for i in poisoned_idx)
            train_poison_labels.extend(labels[i] for i in poisoned_idx)

            if dataset in ('domain', 'office') and args.aug_mult > 1:
                train_images, train_labels = images * args.aug_mult, labels * args.aug_mult
            else:
                train_images, train_labels = images, labels

            train_ds = InMemoryImageDataset(train_images, train_labels, transform=transform_train)
            train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=4)
        else:
            train_dl, _ = get_dataloader(args, dataset=dataset, data_dir=args.data_dir,
                                          train_bs=args.batch_size, test_bs=args.batch_size, dataidxs=dataidxs)
        client2loaders[client_id] = train_dl

    if train_poison_images:
        train_poison_dl = DataLoader(
            InMemoryImageDataset(train_poison_images, train_poison_labels, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        train_poison_dl = None

    # ---- test side: clean test set + every non-target test image triggered ----
    test_images, test_labels = _load_cdls_raw_images(args, dataset, domain, None, train=False)

    clean_test_dl = DataLoader(
        InMemoryImageDataset([im.copy() for im in test_images], list(test_labels), transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    bd_src = [i for i in range(len(test_labels)) if int(test_labels[i]) != int(target_label)]
    if bd_src:
        bd_images = [add_trigger(test_images[i], trigger_size, trigger_value) for i in bd_src]
        bd_labels = [int(target_label)] * len(bd_src)   # ASR = fraction predicted as target
        backdoor_test_dl = DataLoader(
            InMemoryImageDataset(bd_images, bd_labels, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        backdoor_test_dl = None

    return client2loaders, clean_test_dl, backdoor_test_dl, train_poison_dl


def pgd_project_(net, global_vec, eps):

    w = list(net.parameters())
    w_vec = torch.nn.utils.parameters_to_vector(w)
    diff = w_vec - global_vec
    dnorm = torch.norm(diff)
    if dnorm > eps:
        torch.nn.utils.vector_to_parameters(global_vec + eps * diff / dnorm, w)