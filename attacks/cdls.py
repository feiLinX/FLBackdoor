import torch
import random
import os
import sys
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import get_dataloader
from data_aug_utils import AutoAugment
from datasets.crossdomain import DigitsDataset

def _pil_resize_like(img, target_hw):

    if img.shape[:2] == tuple(target_hw):
        return img
    resized = Image.fromarray(img).resize((target_hw[1], target_hw[0]))
    return np.array(resized)


def _kld_raw(img_a, img_b):

    ta = torch.from_numpy(img_a.astype(np.float32))
    tb = torch.from_numpy(img_b.astype(np.float32))

    h1 = torch.clip(ta, 1e-10, None)
    h2 = torch.clip(tb, 1e-10, None)

    h1 = h1.flatten()
    h2 = h2.flatten()

    kld_1 = entropy(h1, h2)
    kld_2 = entropy(h2, h1)
    kld = (kld_1 + kld_2)/2
    return kld


def get_digits_transforms(args):

    normalize = transforms.Normalize((0.1307,), (0.3081,))
    transform_train = [
        transforms.ToPILImage(),
        transforms.Resize((36, 36)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]

    if args.auto_aug:
        transform_train.append(AutoAugment())

    transform_train.extend([
        transforms.ToTensor(),
        normalize,
    ])
    transform_train = transforms.Compose(transform_train)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    return transform_train, transform_test


def build_digits_donor_pool(data_dir, donor_domains, target_label, train=True,
                             pool_size_per_domain=500, seed=0):

    rng = random.Random(seed)
    pool = []
    for domain in donor_domains:
        ds = DigitsDataset(data_dir, domain, train=train, transform=None)
        eligible = [i for i in range(len(ds)) if ds.labels[i] != target_label]
        if len(eligible) > pool_size_per_domain:
            eligible = rng.sample(eligible, pool_size_per_domain)
        for i in eligible:
            img, label = ds[i]
            pool.append((img, label))
    return pool


def poison_label_swap(images, labels, target_label, partition, donor_pool,
                       max_search=200, threshold=None, seed=0):

    rng = random.Random(seed)

    victim_idx = [i for i, l in enumerate(labels) if int(l) == int(target_label)]
    n_replace = int(round(len(victim_idx) * partition))
    victim_idx = rng.sample(victim_idx, n_replace) if n_replace > 0 else []

    replaced = []
    for idx in victim_idx:
        victim_img = images[idx]
        target_hw = victim_img.shape[:2]

        search_pool = donor_pool if len(donor_pool) <= max_search else rng.sample(donor_pool, max_search)

        best_dist, best_img = float('inf'), None
        for donor_img, _ in search_pool:
            donor_resized = _pil_resize_like(donor_img, target_hw)
            dist = _kld_raw(victim_img, donor_resized)
            if dist < best_dist:
                best_dist, best_img = dist, donor_resized

        if best_img is None:
            continue
        if threshold is not None and best_dist >= threshold:
            continue

        images[idx] = best_img  # replace pixels; labels[idx] stays target_label
        replaced.append(idx)
    return replaced


class InMemoryImageDataset(Dataset):

    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
    

def build_digits_backdoor(args, client2dataidx, adv_clients, target_label, partition,
                           domain='mnist', donor_domains=('mnist_m', 'svhn', 'syn', 'usps'),
                           donor_pool_size=500, max_search=200, threshold=None, seed=0):

    transform_train, transform_test = get_digits_transforms(args)

    # ---- train side: poison the selected clients' own-domain data ----
    train_donor_pool = build_digits_donor_pool(args.data_dir, donor_domains, target_label,
                                                train=True, pool_size_per_domain=donor_pool_size, seed=seed)

    client2loaders = {}
    train_poison_images, train_poison_labels = [], []
    for client_id in range(args.nclients):
        dataidxs = client2dataidx[client_id]
        if client_id in adv_clients:
            raw_ds = DigitsDataset(args.data_dir, domain, dataidxs=dataidxs, train=True, transform=None)
            images = [raw_ds[i][0] for i in range(len(raw_ds))]
            labels = [raw_ds[i][1] for i in range(len(raw_ds))]

            replaced_idx = poison_label_swap(images, labels, target_label, partition, train_donor_pool,
                                              max_search=max_search, threshold=threshold, seed=seed + client_id)
            # keep the literal poisoned (image, label) pairs for the train_asr diagnostic
            train_poison_images.extend(images[i] for i in replaced_idx)
            train_poison_labels.extend(labels[i] for i in replaced_idx)

            train_ds = InMemoryImageDataset(images, labels, transform=transform_train)
            train_dl = DataLoader(dataset=train_ds, batch_size=args.batch_size, drop_last=False, shuffle=True, num_workers=4)
        else:
            train_dl, _ = get_dataloader(args, dataset='digits', data_dir=args.data_dir,
                                          train_bs=args.batch_size, test_bs=args.batch_size, dataidxs=dataidxs)
        client2loaders[client_id] = train_dl

    if train_poison_images:
        train_poison_dl = DataLoader(
            InMemoryImageDataset(train_poison_images, train_poison_labels, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        train_poison_dl = None

    # ---- test side: build one mixed test set, split into clean vs replaced ----
    test_donor_pool = build_digits_donor_pool(args.data_dir, donor_domains, target_label,
                                               train=False, pool_size_per_domain=donor_pool_size, seed=seed)

    test_ds_raw = DigitsDataset(args.data_dir, domain, train=False, transform=None)
    test_images = [test_ds_raw[i][0] for i in range(len(test_ds_raw))]
    test_labels = [test_ds_raw[i][1] for i in range(len(test_ds_raw))]

    replaced_idx = poison_label_swap(test_images, test_labels, target_label, partition, test_donor_pool,
                                      max_search=max_search, threshold=threshold, seed=seed + 10_000)
    replaced_set = set(replaced_idx)

    clean_idx = [i for i in range(len(test_labels)) if i not in replaced_set]
    clean_test_dl = DataLoader(
        InMemoryImageDataset([test_images[i] for i in clean_idx], [test_labels[i] for i in clean_idx],
                             transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    if replaced_idx:
        backdoor_test_dl = DataLoader(
            InMemoryImageDataset([test_images[i] for i in replaced_idx], [test_labels[i] for i in replaced_idx],
                                 transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        backdoor_test_dl = None

    return client2loaders, clean_test_dl, backdoor_test_dl, train_poison_dl