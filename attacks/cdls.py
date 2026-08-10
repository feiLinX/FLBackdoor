import random
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from scipy.stats import entropy
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import get_dataloader, TwoCropTransform
from data_aug_utils import AutoAugment
from datasets import DigitsDataset, DomainNetDataset, CIFAR10_truncated


CDLS_CONFIG = {

    'digits': {
        'victim_domain': 'mnist',
        'donor_dataset': 'digits',
        'donor_domains': ['mnist_m', 'svhn', 'syn', 'usps'],
        'exclude_target_label': True,
    },

    'domain': {
        'victim_domain': 'clipart',
        'donor_dataset': 'domain',
        'donor_domains': ['infograph', 'painting', 'quickdraw', 'real', 'sketch'],
        'exclude_target_label': True,
    },

    'cifar10': {
        'victim_domain': None,
        'donor_dataset': 'digits',
        'donor_domains': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'exclude_target_label': False,
    },
}


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


def _kld_vec(p, q):
    """Symmetric KL (Jeffreys / 2) between two 1-D probability-like vectors."""
    p = torch.clip(p.float(), 1e-10, None); p = p / p.sum()
    q = torch.clip(q.float(), 1e-10, None); q = q / q.sum()
    kl_pq = torch.sum(p * (p.log() - q.log()))
    kl_qp = torch.sum(q * (q.log() - p.log()))

    return (0.5 * (kl_pq + kl_qp)).item()


def _compute_donor_reprs(donor_pool, distance_mode, extractor):
    """Precompute donor representations for embed_kl / pred_kl (None for raw_kl)."""
    if distance_mode == 'raw_kl' or extractor is None:
        return None
    imgs = [img for img, _ in donor_pool]
    if distance_mode == 'embed_kl':
        return F.softmax(extractor.embed(imgs), dim=1)
    elif distance_mode == 'pred_kl':
        return extractor.predict(imgs)
    else:
        raise ValueError("unknown distance_mode: %s" % distance_mode)


def _make_donor_dataset(data_dir, donor_dataset, domain, train):
    """Instantiate the raw (transform=None) dataset a donor domain is drawn from."""
    if donor_dataset == 'digits':
        return DigitsDataset(data_dir, domain, train=train, transform=None)
    elif donor_dataset == 'domain':
        return DomainNetDataset(data_dir, domain, train=train, transform=None)
    else:
        raise ValueError("unknown donor_dataset: %s" % donor_dataset)


def _load_cdls_raw_images(args, dataset, domain, dataidxs, train=True):

    if dataset == 'digits':
        ds = DigitsDataset(args.data_dir, domain, dataidxs=dataidxs, train=train, transform=None)
    elif dataset == 'domain':
        ds = DomainNetDataset(args.data_dir, domain, dataidxs=dataidxs, train=train, transform=None)
    elif dataset == 'cifar10':
        ds = CIFAR10_truncated(args.data_dir, dataidxs=dataidxs, train=train, transform=None, download=True)
    else:
        raise ValueError("CDLS not supported for dataset: %s" % dataset)
    images = [ds[i][0] for i in range(len(ds))]
    labels = [int(ds[i][1]) for i in range(len(ds))]

    return images, labels


# ---------------------------- Embedding-KLD and Prediction-KLD ----------------------------
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

class SimCLRNet(nn.Module):
    """CIFAR-style ResNet18 encoder f (32x32 stem) + MLP projection head g."""
    def __init__(self, proj_dim=128):
        super().__init__()
        base = torchvision.models.resnet18()
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(base.conv1.weight, mode='fan_out', nonlinearity='relu')
        base.maxpool = nn.Identity()
        self.encoder = nn.Sequential(*list(base.children())[:-1], nn.Flatten())  # -> [B, 512]
        self.feat_dim = 512
        self.projector = ProjectionHead(self.feat_dim, self.feat_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)      # representation used for embed_kl / pred_kl
        z = self.projector(h)    # used only by the contrastive loss
        return h, z

class SimCLRResNet(nn.Module):
    """ImageNet-stem ResNet18 encoder f + MLP projection head g for natural images."""
    def __init__(self, proj_dim=128):
        super().__init__()
        base = torchvision.models.resnet18()
        self.feat_dim = base.fc.in_features            # 512
        base.fc = nn.Identity()
        self.encoder = base                            # -> [B, 512]
        self.projector = ProjectionHead(self.feat_dim, self.feat_dim, proj_dim)

    def forward(self, x):
        h = self.encoder(x)      # representation used for embed_kl / pred_kl
        z = self.projector(h)    # used only by the contrastive loss
        return h, z


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
    

def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent (SimCLR) loss over 2B views; z1[i] and z2[i] are a positive pair."""
    B = z1.size(0)
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # [2B, D]
    sim = (z @ z.t()) / temperature                      # [2B, 2B]
    self_mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(self_mask, float('-inf'))
    targets = (torch.arange(2 * B, device=z.device) + B) % (2 * B)

    return F.cross_entropy(sim, targets)


def get_simclr_transform():

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(32, scale=(0.5, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])


def get_simclr_transform_domain(img_size=96):

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5)], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])



def pretrain_simclr_digits(args, domains, epochs=None, proj_dim=None,
                           batch_size=None, temperature=None, lr=1e-3, logger=None):

    epochs = args.bd_simclr_epochs if epochs is None else epochs
    proj_dim = args.bd_simclr_dim if proj_dim is None else proj_dim
    batch_size = args.bd_simclr_bs if batch_size is None else batch_size
    temperature = args.bd_simclr_temp if temperature is None else temperature

    two_crop = TwoCropTransform(get_simclr_transform())
    ds_list = [DigitsDataset(args.data_dir, d, train=True, transform=two_crop) for d in domains]
    pool = torch.utils.data.ConcatDataset(ds_list)
    loader = DataLoader(pool, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SimCLRNet(proj_dim=proj_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for epoch in range(epochs):
        running, nb = 0.0, 0
        for views, _ in loader:
            v1, v2 = views[0].cuda(), views[1].cuda()
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
        msg = 'SimCLR pretrain epoch %d/%d | loss %.4f' % (epoch + 1, epochs, running / max(nb, 1))
        if logger is not None:
            logger.info(msg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(msg)
    model.eval()
    return model


def pretrain_simclr_domain(args, domains, img_size=96, epochs=None, proj_dim=None,
                           batch_size=None, temperature=None, lr=1e-3, logger=None):

    epochs = args.bd_simclr_epochs if epochs is None else epochs
    proj_dim = args.bd_simclr_dim if proj_dim is None else proj_dim
    batch_size = args.bd_simclr_bs if batch_size is None else batch_size
    temperature = args.bd_simclr_temp if temperature is None else temperature

    two_crop = TwoCropTransform(get_simclr_transform_domain(img_size))
    ds_list = [DomainNetDataset(args.data_dir, d, train=True, transform=two_crop) for d in domains]
    pool = torch.utils.data.ConcatDataset(ds_list)
    loader = DataLoader(pool, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    model = SimCLRResNet(proj_dim=proj_dim).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    model.train()
    for epoch in range(epochs):
        running, nb = 0.0, 0
        for views, _ in loader:
            v1, v2 = views[0].cuda(), views[1].cuda()
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = nt_xent_loss(z1, z2, temperature)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
        msg = 'SimCLR(domain) pretrain epoch %d/%d | loss %.4f' % (epoch + 1, epochs, running / max(nb, 1))
        if logger is not None:
            logger.info(msg)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(msg)
    model.eval()

    return model


def train_adv_classifier(encoder, args, domains, epochs=10, lr=1e-2, logger=None):

    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    ds_list = [DigitsDataset(args.data_dir, d, train=True, transform=tf) for d in domains]
    pool = torch.utils.data.ConcatDataset(ds_list)
    loader = DataLoader(pool, batch_size=256, shuffle=True, num_workers=4)

    encoder = encoder.cuda().eval()
    clf = nn.Linear(encoder.feat_dim, 10).cuda()
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    clf.train()
    for epoch in range(epochs):
        running, nb = 0.0, 0
        for x, y in loader:
            x, y = x.cuda(), y.to(dtype=torch.int64).cuda()
            with torch.no_grad():
                h = encoder.encoder(x)
            loss = criterion(clf(h), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
        if logger is not None:
            logger.info('adv-classifier epoch %d/%d | loss %.4f' % (epoch + 1, epochs, running / max(nb, 1)))
    clf.eval()

    return clf


def train_adv_classifier_domain(encoder, args, domains, img_size=96, epochs=10, lr=1e-2, logger=None):
    """Linear probe (adversary classifier) on the frozen DomainNet SimCLR encoder;
    used by pred_kl to obtain class-probability distributions for donor selection."""
    tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds_list = [DomainNetDataset(args.data_dir, d, train=True, transform=tf) for d in domains]
    pool = torch.utils.data.ConcatDataset(ds_list)
    loader = DataLoader(pool, batch_size=128, shuffle=True, num_workers=4)

    encoder = encoder.cuda().eval()
    clf = nn.Linear(encoder.feat_dim, 10).cuda()
    optimizer = optim.Adam(clf.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    clf.train()
    for epoch in range(epochs):
        running, nb = 0.0, 0
        for x, y in loader:
            x, y = x.cuda(), y.to(dtype=torch.int64).cuda()
            with torch.no_grad():
                h = encoder.encoder(x)
            loss = criterion(clf(h), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item(); nb += 1
        if logger is not None:
            logger.info('adv-classifier(domain) epoch %d/%d | loss %.4f' % (epoch + 1, epochs, running / max(nb, 1)))
    clf.eval()

    return clf


class AdversaryExtractor:

    def __init__(self, encoder, classifier=None, img_size=32, normalize=None):
        self.encoder = encoder.cuda().eval()
        self.classifier = classifier.cuda().eval() if classifier is not None else None
        if normalize is None:
            normalize = transforms.Normalize((0.1307,), (0.3081,))  # digit default
        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])

    def _to_batch(self, images):
        return torch.stack([self.tf(im) for im in images]).cuda()

    @torch.no_grad()
    def embed(self, images, bs=256):
        outs = []
        for i in range(0, len(images), bs):
            outs.append(self.encoder.encoder(self._to_batch(images[i:i + bs])).cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty(0)

    @torch.no_grad()
    def predict(self, images, bs=256):
        assert self.classifier is not None, "pred_kl needs an adversary classifier"
        outs = []
        for i in range(0, len(images), bs):
            h = self.encoder.encoder(self._to_batch(images[i:i + bs]))
            outs.append(F.softmax(self.classifier(h), dim=1).cpu())

        return torch.cat(outs, dim=0) if outs else torch.empty(0)


def build_donor_pool(data_dir, donor_domains, target_label, train=True,
                     pool_size_per_domain=500, seed=0,
                     donor_dataset='digits', exclude_target_label=True):

    rng = random.Random(seed)
    pool = []
    for domain in donor_domains:
        ds = _make_donor_dataset(data_dir, donor_dataset, domain, train)
        if exclude_target_label:
            eligible = [i for i in range(len(ds)) if ds.labels[i] != target_label]
        else:
            eligible = list(range(len(ds)))
        if len(eligible) > pool_size_per_domain:
            eligible = rng.sample(eligible, pool_size_per_domain)
        for i in eligible:
            img, label = ds[i]
            pool.append((img, label))

    return pool


def poison_label_swap(images, labels, target_label, partition, donor_pool,
                       max_search=200, threshold=None, seed=0,
                       distance_mode='raw_kl', extractor=None, donor_reprs=None):

    rng = random.Random(seed)

    victim_idx = [i for i, l in enumerate(labels) if int(l) == int(target_label)]
    n_replace = int(round(len(victim_idx) * partition))
    victim_idx = rng.sample(victim_idx, n_replace) if n_replace > 0 else []

    # precompute victim representations once (embed_kl / pred_kl only)
    victim_reprs = None
    if distance_mode in ('embed_kl', 'pred_kl') and victim_idx:
        victim_imgs = [images[i] for i in victim_idx]
        if distance_mode == 'embed_kl':
            victim_reprs = F.softmax(extractor.embed(victim_imgs), dim=1)
        else:
            victim_reprs = extractor.predict(victim_imgs)

    pool_size = len(donor_pool)
    replaced = []
    for vpos, idx in enumerate(victim_idx):
        victim_img = images[idx]
        target_hw = victim_img.shape[:2]

        if pool_size <= max_search:
            cand = range(pool_size)
        else:
            cand = rng.sample(range(pool_size), max_search)

        best_dist, best_j = float('inf'), None
        if distance_mode == 'raw_kl':
            for j in cand:
                donor_resized = _pil_resize_like(donor_pool[j][0], target_hw)
                dist = _kld_raw(victim_img, donor_resized)
                if dist < best_dist:
                    best_dist, best_j = dist, j
        else:
            vr = victim_reprs[vpos]
            for j in cand:
                dist = _kld_vec(vr, donor_reprs[j])
                if dist < best_dist:
                    best_dist, best_j = dist, j

        if best_j is None:
            continue
        if threshold is not None and best_dist >= threshold:
            continue

        # replace pixels with the (resized) donor image; labels[idx] stays target_label
        images[idx] = _pil_resize_like(donor_pool[best_j][0], target_hw)
        replaced.append(idx)

    return replaced


def get_cdls_transforms(args, dataset):

    if dataset == 'digits':
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
    
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        tt = [transforms.ToPILImage(),
              transforms.RandomCrop(32, padding=4),
              transforms.RandomHorizontalFlip()]
        if args.auto_aug:
            tt.append(AutoAugment())
        tt += [transforms.ToTensor(), normalize]
        return transforms.Compose(tt), transforms.Compose([transforms.ToTensor(), normalize])
    
    elif dataset == 'domain':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        tt = [transforms.ToPILImage(),
              transforms.Resize((256, 256)),
              transforms.RandomCrop(224),
              transforms.RandomHorizontalFlip()]
        if args.auto_aug:
            tt.append(AutoAugment())
        tt += [transforms.ToTensor(), normalize]
        transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(), normalize])
        return transforms.Compose(tt), transform_test
    
    else:
        raise ValueError("CDLS not supported for dataset: %s" % dataset)



def build_cdls_backdoor(args, client2dataidx, adv_clients, target_label, partition,
                        dataset='digits', domain=None, donor_dataset=None, donor_domains=None,
                        donor_pool_size=500, max_search=200, threshold=None, seed=0,
                        distance_mode='raw_kl', extractor=None):

    cfg = CDLS_CONFIG[dataset]
    domain = cfg['victim_domain'] if domain is None else domain
    donor_dataset = cfg['donor_dataset'] if donor_dataset is None else donor_dataset
    donor_domains = cfg['donor_domains'] if donor_domains is None else donor_domains
    exclude_target = cfg['exclude_target_label']

    transform_train, transform_test = get_cdls_transforms(args, dataset)

    # ---- train side: poison the selected clients' own-domain data ----
    train_donor_pool = build_donor_pool(args.data_dir, donor_domains, target_label,
                                        train=True, pool_size_per_domain=donor_pool_size, seed=seed,
                                        donor_dataset=donor_dataset, exclude_target_label=exclude_target)
    train_donor_reprs = _compute_donor_reprs(train_donor_pool, distance_mode, extractor)

    client2loaders = {}
    train_poison_images, train_poison_labels = [], []
    for client_id in range(args.nclients):
        dataidxs = client2dataidx[client_id]
        if client_id in adv_clients:
            images, labels = _load_cdls_raw_images(args, dataset, domain, dataidxs, train=True)

            replaced_idx = poison_label_swap(images, labels, target_label, partition, train_donor_pool,
                                             max_search=max_search, threshold=threshold, seed=seed + client_id,
                                             distance_mode=distance_mode, extractor=extractor,
                                             donor_reprs=train_donor_reprs)
            # keep the literal poisoned (image, label) pairs for the train_asr diagnostic
            train_poison_images.extend(images[i] for i in replaced_idx)
            train_poison_labels.extend(labels[i] for i in replaced_idx)

            # DomainNet/Office benign clients get aug_mult-tiled loaders (see
            # get_dataloader); tile the malicious client too so its per-round
            # dataset size (and thus FedAvg weight) matches the benign clients.
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

    # ---- test side: build one mixed test set, split into clean vs replaced ----
    test_donor_pool = build_donor_pool(args.data_dir, donor_domains, target_label,
                                       train=False, pool_size_per_domain=donor_pool_size, seed=seed,
                                       donor_dataset=donor_dataset, exclude_target_label=exclude_target)
    test_donor_reprs = _compute_donor_reprs(test_donor_pool, distance_mode, extractor)

    test_images, test_labels = _load_cdls_raw_images(args, dataset, domain, None, train=False)

    test_partition = 0.5 if dataset == 'domain' else partition

    replaced_idx = poison_label_swap(test_images, test_labels, target_label, test_partition, test_donor_pool,
                                      max_search=max_search, threshold=threshold, seed=seed + 10_000,
                                      distance_mode=distance_mode, extractor=extractor,
                                      donor_reprs=test_donor_reprs)
    replaced_set = set(replaced_idx)

    clean_idx = [i for i in range(len(test_labels)) if i not in replaced_set]
    clean_imgs, clean_lbls = [test_images[i] for i in clean_idx], [test_labels[i] for i in clean_idx]
    bd_imgs, bd_lbls = [test_images[i] for i in replaced_idx], [test_labels[i] for i in replaced_idx]

    # DomainNet/Office: tile the raw test set ×aug_mult, mirroring the train-side aug_mult
    if dataset in ('domain', 'office') and args.aug_mult > 1:
        clean_imgs, clean_lbls = clean_imgs * args.aug_mult, clean_lbls * args.aug_mult
        bd_imgs, bd_lbls = bd_imgs * args.aug_mult, bd_lbls * args.aug_mult

    clean_test_dl = DataLoader(
        InMemoryImageDataset(clean_imgs, clean_lbls, transform=transform_test),
        batch_size=args.batch_size, shuffle=False, num_workers=4)

    if bd_imgs:
        backdoor_test_dl = DataLoader(
            InMemoryImageDataset(bd_imgs, bd_lbls, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        backdoor_test_dl = None

    return client2loaders, clean_test_dl, backdoor_test_dl, train_poison_dl



def apply_model_poison_constraint(global_net, nets_current, adv_clients, scale=1.0):

    adv_clients = set(adv_clients)
    g = {k: v.detach().clone().float().cuda() for k, v in global_net.state_dict().items()}
    float_keys = [k for k in g if torch.is_floating_point(g[k])]

    def update_norm(net):
        sd = net.state_dict()
        return torch.sqrt(sum(((sd[k].float().cuda() - g[k]) ** 2).sum() for k in float_keys))

    benign = [cid for cid in nets_current if cid not in adv_clients]
    ref = torch.median(torch.stack([update_norm(nets_current[c]) for c in benign])) if benign else None

    for cid in nets_current:
        if cid not in adv_clients:
            continue
        net = nets_current[cid]
        sd = net.state_dict()
        new_sd = {}
        for k in sd:
            if k in float_keys:
                new_sd[k] = g[k] + scale * (sd[k].float().cuda() - g[k])
            else:
                new_sd[k] = sd[k]
        if ref is not None:
            cur = torch.sqrt(sum(((new_sd[k] - g[k]) ** 2).sum() for k in float_keys))
            if cur > ref:
                f = ref / (cur + 1e-12)
                for k in float_keys:
                    new_sd[k] = g[k] + f * (new_sd[k] - g[k])
        net.load_state_dict({k: new_sd[k].to(sd[k].dtype).to(sd[k].device) for k in sd})
