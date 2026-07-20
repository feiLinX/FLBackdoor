import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data_aug_utils import AutoAugment
from datasets import DigitsDataset, DomainNetDataset, OfficeDataset, CIFAR10_truncated, CIFAR100_truncated


def load_digits_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_trainset = DigitsDataset(data_dir, 'mnist', train=True, transform=transform)
    mnist_testset = DigitsDataset(data_dir, 'mnist', train=False, transform=transform)

    mnistm_trainset = DigitsDataset(data_dir, 'mnist_m', train=True, transform=transform)
    mnistm_testset = DigitsDataset(data_dir, 'mnist_m', train=False, transform=transform)

    svhn_trainset = DigitsDataset(data_dir, 'svhn', train=True, transform=transform)
    svhn_testset = DigitsDataset(data_dir, 'svhn', train=False, transform=transform)

    syn_trainset = DigitsDataset(data_dir, 'syn', train=True, transform=transform)
    syn_testset = DigitsDataset(data_dir, 'syn', train=False, transform=transform)

    usps_trainset = DigitsDataset(data_dir, 'usps', train=True, transform=transform)
    usps_testset = DigitsDataset(data_dir, 'usps', train=False, transform=transform)

    train_dict = {'mnist': mnist_trainset, 'mnist_m': mnistm_trainset, 'svhn': svhn_trainset, 
                  'syn': syn_trainset, 'usps': usps_trainset}

    test_dict = {'mnist': mnist_testset, 'mnist_m': mnistm_testset, 'svhn': svhn_testset,
        'syn': syn_testset, 'usps': usps_testset}

    return train_dict, test_dict

    
def load_domain_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    
    clipart_trainset = DomainNetDataset(data_dir, 'clipart', train=True, transform=transform)
    clipart_testset = DomainNetDataset(data_dir, 'clipart', train=False, transform=transform)

    infograph_trainset = DomainNetDataset(data_dir, 'infograph', train=True, transform=transform)
    infograph_testset = DomainNetDataset(data_dir, 'infograph', train=False, transform=transform)

    painting_trainset = DomainNetDataset(data_dir, 'painting', train=True, transform=transform)
    painting_testset = DomainNetDataset(data_dir, 'painting', train=False, transform=transform)

    quickdraw_trainset = DomainNetDataset(data_dir, 'quickdraw', train=True, transform=transform)
    quickdraw_testset = DomainNetDataset(data_dir, 'quickdraw', train=False, transform=transform)

    real_trainset = DomainNetDataset(data_dir, 'real', train=True, transform=transform)
    real_testset = DomainNetDataset(data_dir, 'real', train=False, transform=transform)
    
    sketch_trainset = DomainNetDataset(data_dir, 'sketch', train=True, transform=transform)
    sketch_testset = DomainNetDataset(data_dir, 'sketch', train=False, transform=transform)

    train_dict = {'clipart': clipart_trainset, 'infograph': infograph_trainset, 'painting': painting_trainset,
        'quickdraw': quickdraw_trainset, 'real': real_trainset, 'sketch': sketch_trainset}

    test_dict = {'clipart': clipart_testset, 'infograph': infograph_testset, 'painting': painting_testset,
        'quickdraw': quickdraw_testset, 'real': real_testset, 'sketch': sketch_testset}

    return train_dict, test_dict


def load_office_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])
    
    amazon_trainset = OfficeDataset(data_dir, 'amazon', train=True, transform=transform)
    amazon_testset = OfficeDataset(data_dir, 'amazon', train=False, transform=transform)

    caltech_trainset = OfficeDataset(data_dir, 'caltech', train=True, transform=transform)
    caltech_testset = OfficeDataset(data_dir, 'caltech', train=False, transform=transform)

    dslr_trainset = OfficeDataset(data_dir, 'dslr', train=True, transform=transform)
    dslr_testset = OfficeDataset(data_dir, 'dslr', train=False, transform=transform)

    webcam_trainset = OfficeDataset(data_dir, 'webcam', train=True, transform=transform)
    webcam_testset = OfficeDataset(data_dir, 'webcam', train=False, transform=transform)

    train_dict = {'amazon': amazon_trainset, 'dslr': dslr_trainset, 'webcam': webcam_trainset, 'caltech': caltech_trainset}
    test_dict = {'amazon': amazon_testset, 'dslr': dslr_testset, 'webcam': webcam_testset, 'caltech': caltech_testset}

    return train_dict, test_dict


def load_cifar10_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar10_train_ds = CIFAR10_truncated(data_dir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(data_dir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)


def load_cifar100_data(data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    cifar100_train_ds = CIFAR100_truncated(data_dir, train=True, download=True, transform=transform)
    cifar100_test_ds = CIFAR100_truncated(data_dir, train=False, download=True, transform=transform)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target

    return (X_train, y_train, X_test, y_test)


def partition_data(dataset, datadir, partition, n_clients, alpha=0.4):
    if dataset == 'cifar10':
        X_train, y_train, X_test, y_test = load_cifar10_data(datadir)
    elif dataset == 'cifar100':
        X_train, y_train, X_test, y_test = load_cifar100_data(datadir)
    elif dataset == 'digits':
        dict_train, dict_test = load_digits_data(datadir)
        y_train = np.array(dict_train['mnist'].labels)
    elif dataset == 'domain':
        dict_train, dict_test = load_domain_data(datadir)
        y_train = np.array(dict_train['clipart'].labels)
    elif dataset == 'office':
        dict_train, dict_test = load_office_data(datadir)
        y_train = np.array(dict_train['caltech'].labels)
    else:
        raise NotImplementedError("dataset not imeplemented")

    n_train = y_train.shape[0]

    if partition == "homo" or partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_clients)
        client2dataidx = {i: batch_idxs[i] for i in range(n_clients)}

    elif partition == "noniid-labeldir" or partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10
        if dataset == 'cifar100':
            K = 100

        N = y_train.shape[0]
        client2dataidx = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_clients)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
                proportions = np.array([p * (len(idx_j) < N / n_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(n_clients):
            np.random.shuffle(idx_batch[j])
            client2dataidx[j] = idx_batch[j]

    return client2dataidx


def get_dataloader(args, dataset, data_dir, train_bs, test_bs, dataidxs=None):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated

        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        transform_train = [
            transforms.ToPILImage(),
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

        train_ds = dl_obj(data_dir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(data_dir, train=False, transform=transform_test, download=True)

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=4)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4)

    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated

        normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                        std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
        transform_train = [
            transforms.ToPILImage(),
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
            normalize
        ])

        train_ds = dl_obj(data_dir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
        test_ds = dl_obj(data_dir, train=False, transform=transform_test, download=True)

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=4)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4)
    
    elif dataset == 'digits':
        dl_obj = DigitsDataset
        subset = 'mnist'

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
        
        #transform_train, transform_test = get_digits_transforms(args)

        train_ds = dl_obj(data_dir, subset, dataidxs=dataidxs, train=True, transform=transform_train)
        test_ds = dl_obj(data_dir, subset, train=False, transform=transform_test)

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=4)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4)

    elif dataset == 'domain':
        dl_obj = DomainNetDataset
        subset = 'clipart'

        # ImageNet mean/std, since DomainNet consists of natural images and models
        # here (e.g., ResNet) are typically pretrained on ImageNet
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_train = [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
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
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        if dataidxs is not None and args.aug_mult > 1:
            # Replicate the client's indices so the same underlying images are
            # sampled (and independently transformed via the random augmentations
            # above) multiple times per epoch. This is what actually increases
            # the number of samples/batches a client trains on per round --
            # the augmentation transforms alone only diversify existing samples
            # in place without changing len(dataset). Only applied for 'domain'
            # and 'office', whose clients have far fewer raw images per class.
            dataidxs = np.tile(np.asarray(dataidxs), args.aug_mult)

        train_ds = dl_obj(data_dir, subset, dataidxs=dataidxs, train=True, transform=transform_train)
        test_ds = dl_obj(data_dir, subset, train=False, transform=transform_test)

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=4)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4)

    elif dataset == 'office':
        dl_obj = OfficeDataset
        subset = 'caltech'

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform_train = [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
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
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

        if dataidxs is not None and args.aug_mult > 1:
            dataidxs = np.tile(np.asarray(dataidxs), args.aug_mult)

        train_ds = dl_obj(data_dir, subset, dataidxs=dataidxs, train=True, transform=transform_train)
        test_ds = dl_obj(data_dir, subset, train=False, transform=transform_test)

        train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, drop_last=False, shuffle=True, num_workers=4)
        test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, num_workers=4)

    return train_dl, test_dl