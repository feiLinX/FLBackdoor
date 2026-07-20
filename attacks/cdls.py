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
    """Resize a uint8 HWC numpy image to match target_hw=(H, W) if needed,
    so cross-domain donor images (which may come from a differently-sized
    digit-5 domain) can replace a victim sample of another domain's shape.
    """
    if img.shape[:2] == tuple(target_hw):
        return img
    resized = Image.fromarray(img).resize((target_hw[1], target_hw[0]))
    return np.array(resized)


def _kld_raw(img_a, img_b):
    """KL_distance between two uint8 HWC numpy images, treating raw pixels as the feature."""
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
    """Build the (train, test) transform pipelines used for the 'digits' dataset.
    Pulled out of get_dataloader so the backdoor pipeline can reuse the exact same
    augmentation/normalization when wrapping in-memory (poisoned) client data.
    """
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
    """Collect (uint8 HWC image, int label) samples from `donor_domains` (any
    digit-5 sub-dataset other than the victim client's own domain) whose
    label != target_label -- the only samples eligible to be planted in
    place of a target_label sample. `pool_size_per_domain` randomly
    subsamples each domain to keep the nearest-neighbor search tractable.
    """
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
    """Replace `partition` fraction of `images`/`labels` entries whose label
    equals `target_label` with the visually-nearest (raw-pixel KL divergence)
    sample from `donor_pool` (a list of (image, label) pairs whose label is
    guaranteed != target_label). The replaced entry's label is left as
    `target_label`, so a model trained on it associates the donor's visual
    pattern with `target_label` instead of the donor's own true label --
    this is the backdoor.

    Args:
        images, labels: parallel lists; `images` entries are replaced in place.
        target_label:   the label whose samples are targeted for replacement.
        partition:      fraction in [0, 1] of target_label samples to replace.
        donor_pool:     list of (image, label) candidates, label != target_label.
        max_search:     cap on how many donor_pool entries are scanned per
                        victim sample (keeps the search tractable).
        threshold:      if set, only replace when the best KL distance found
                        is below this value; None (default) always uses the
                        best match found.
        seed:           for reproducible victim/donor sampling.

    Returns:
        list of indices (into images/labels) that were actually replaced.
    """
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
    """Wrap a list of (uint8 HWC numpy image, int label) pairs so poisoned
    data -- which mixes raw pixels sourced from multiple digit-domain
    donors -- can go through the same transform pipeline as the path-based
    DigitsDataset (ToPILImage -> ... -> ToTensor -> Normalize).
    """
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
    """Build client train loaders (with `adv_clients` poisoned) plus the
    loaders needed to report ACC/ASR/train_asr for the cross-domain
    label-swap backdoor.

    - `adv_clients`: list of client ids to poison -- its length is the
      "number of clients" running the attack.
    - `target_label`: the "original" label whose samples get replaced.
    - `partition`: fraction of that client's target_label samples to replace.

    Returns:
        client2loaders:   dict[client_id -> DataLoader]; benign clients use
            the normal path-based pipeline, adv_clients use an in-memory
            poisoned dataset.
        clean_test_dl:    DataLoader over the untouched (held-out) test
            samples -> ACC.
        backdoor_test_dl: DataLoader over freshly-poisoned held-out test
            samples (labeled target_label) -> ASR, which measures whether
            the backdoor generalizes to unseen donor-style inputs. None if
            no test sample was replaced.
        train_poison_dl:  DataLoader over the literal poisoned training
            samples planted into adv_clients (labeled target_label) ->
            train_asr, a memorization diagnostic (see evaluate_acc_asr).
            None if nothing was replaced.
    """
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