import os, sys, copy, glob
import time, random
import numpy as np
import pickle
import logging
import datetime
import json
import gc
import argparse

import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import torchvision
import torchvision.transforms as transforms

from utils import *
from data_utils import *
from models.resnets import ResNet18, ResNet34, ResNet50, ResNet18Small, ResNet34Small, ResNet50Small
from models.mobilenetv2 import MobileNetV2, MobileNetV2Large
from models.cnns import FangCNN
from attacks.cdls import build_digits_backdoor
from data_aug_utils import AutoAugment
from aggregations.fedavg import fedavg_local, fedavg_global


def args_parser():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--dataset", help="dataset", default='digits', type=str,
                        choices=['digits', 'office', 'domain', 'cifar10', 'cifar100'])
    parser.add_argument("--model", help="training model", default="resnet18", type=str,
                        choices=['cnn','resnet18', 'resnet34', 'resnet50', 'mobilenetv2'])
    parser.add_argument("--lr", help="learning rate", default=5e-4, type=float)
    parser.add_argument("--momentum", help="SGD momentum", default=0.9, type=float)
    parser.add_argument("--wd", help="weight decay", default=1e-5, type=float)
    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument('--device', help="cpu, cuda", default="cuda", type=str)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)

    # FL
    parser.add_argument("--aggregation", help="aggregation rule", default='fedavg', type=str,
                        choices=['fedavg', 'krum', 'flame'])
    parser.add_argument("--nrounds", help="# global rounds", default=100, type=int)
    parser.add_argument("--epochs", help="# local epochs", default=5, type=int)
    parser.add_argument("--nclients", help="# clients", default=20, type=int)
    parser.add_argument("--fraction", help="fraction of clients", default=1.0, type=float)
    parser.add_argument("--bias", help="degree of label non-iidness", default=1, type=float)
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy, iid or noniid')

    parser.add_argument('--auto_aug', action='store_true', help='whether to apply auto augmentation')
    parser.add_argument('--aug_mult', help="replicate each client's assigned sample indices this many times "
                        "before building the train Dataset, so random augmentations (crop/flip/autoaug) are "
                        "applied to independently-sampled copies each epoch, inflating the effective per-round "
                        "dataset size without adding new raw images", default=10, type=int)

    # Adversarial
    parser.add_argument("--adv_type", help="adv type", default='None', type=str,
                        choices=['None', 'CDLS'])
    parser.add_argument("--nbyz", help="# byzantines / # adversarial clients", default=4, type=int)
    parser.add_argument("--feature", help="feature extraction", default='raw', type=str,
                        choices=['raw','tsne','proto'])
    parser.add_argument("--bd_target_label", help="original label targeted by the CDLS backdoor", default=0, type=int)
    parser.add_argument("--bd_partition", help="fraction of a client's target_label samples to replace with the nearest cross-domain donor sample", default=0.5, type=float)
    parser.add_argument("--bd_adv_clients", help="explicit client ids running the CDLS backdoor; defaults to the first --nbyz clients when not set", default=None, type=int, nargs='+')
    parser.add_argument("--bd_domain", help="digits sub-dataset the clients are assigned to", default='mnist', type=str,
                        choices=['mnist', 'mnist_m', 'svhn', 'syn', 'usps'])
    parser.add_argument("--bd_donor_domains", help="digits sub-datasets donor replacement samples are drawn from; defaults to all domains other than --bd_domain", default=None, type=str, nargs='+')
    parser.add_argument("--bd_donor_pool_size", help="max donor samples per domain kept for the nearest-neighbor search", default=1000, type=int)
    parser.add_argument("--bd_max_search", help="max donor pool entries scanned per victim sample when finding the nearest match", default=500, type=int)
    
    # Logging
    parser.add_argument("--data_dir", type=str, required=False, default="/scratch/jmh8504/data/", 
                        choices=['/scratch/jmh8504/data/', '/export/home/jmh8504/data/'],)

    parser.add_argument('--logdir', type=str, required=False, default="/scratch/jmh8504/FL/flbackdoor/logs/",
                        choices=['/scratch/jmh8504/FL/flbackdoor/logs/', '/export/home/jmh8504/FL/flbackdoor/logs/'],)
                        
    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--ckptdir', type=str, required=False, default="/scratch/jmh8504/FL/flbackdoor/saved_models/",
                        choices=['/scratch/jmh8504/FL/flbackdoor/saved_models/', '/export/home/jmh8504/FL/flbackdoor/saved_models/'],)
    
    parser.add_argument('--print_interval', type=int, default=10,
                        help='how many comm round to print results on screen')
    parser.add_argument('--save_interval', type=int, default=10,

                        help='how many rounds do we save the checkpoint one time') 

    args, unknown = parser.parse_known_args() 

    return args
        

def init_model(nclients, args):
    nets = {net_i: None for net_i in range(nclients)}
    small_input = args.dataset in ['digits', 'cifar10', 'cifar100']
    for net_i in range(nclients):
        if args.model == 'cnn':
            net = FangCNN()
            net.apply(net.init_xavier)
        elif args.model == 'resnet18':
            net = ResNet18Small() if small_input else ResNet18()
        elif args.model == 'resnet34':
            net = ResNet34Small() if small_input else ResNet34()
        elif args.model == 'resnet50':
            net = ResNet50Small() if small_input else ResNet50()
        elif args.model == 'mobilenetv2':
            net = MobileNetV2() if small_input else MobileNetV2Large()
        else:
            raise NotImplementedError("model not implemented")
        nets[net_i] = net
    
    return nets


if __name__ == "__main__":
    args = args_parser()
    print(args)
    #=============== Logging setup ===============
    mkdirs(args.logdir)
    mkdirs(args.ckptdir)
    mkdirs(os.path.join(args.ckptdir, args.aggregation))

    if args.log_file_name is None:
        argument_path = 'experiment_arguments-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = 'experiment_arguments-%s' % args.log_file_name

    argument_path = argument_path + '.json'

    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = 'experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))

    log_path = args.log_file_name + '.log'
    print('log path: ', log_path)

    for handler in logging.root.handlers[:]:
        handler.close()
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=os.path.join(args.logdir, log_path),
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M', level=logging.INFO, filemode='w')

    logger = logging.getLogger()
    #=============== Logging setup ===============

    #=============== Dataset setup ===============
    logger.info("Partitioning data")
    seed_everything(args.init_seed)

    client2dataidx = partition_data(dataset=args.dataset, datadir=args.data_dir, partition=args.partition,
                                     n_clients=args.nclients, alpha=args.bias)
    
    if args.adv_type == 'CDLS':
        adv_clients = args.bd_adv_clients if args.bd_adv_clients is not None else list(range(args.nbyz))
        donor_domains = args.bd_donor_domains if args.bd_donor_domains is not None else \
            [d for d in ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'] if d != args.bd_domain]

        logger.info("Building cross-domain label-swap backdoor (target_label=%s, partition=%s, adv_clients=%s)"
                    % (str(args.bd_target_label), str(args.bd_partition), str(adv_clients)))
    
        # Build poisoned train loaders for selected clients
        client2loaders, test_dl, backdoor_test_dl, train_poison_dl = build_digits_backdoor(
            args, client2dataidx, adv_clients, args.bd_target_label, args.bd_partition,
            domain=args.bd_domain, donor_domains=donor_domains,
            donor_pool_size=args.bd_donor_pool_size, max_search=args.bd_max_search, seed=args.init_seed)
        
        global_train_dl, _ = get_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                         train_bs=args.batch_size, test_bs=args.batch_size)
    elif args.adv_type == 'None':
        logger.info("No attack selected; training on clean data only")

        client2loaders = {}
        for client_id in range(args.nclients):
            train_dl_local, _ = get_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                            train_bs=args.batch_size, test_bs=args.batch_size, dataidxs=client2dataidx[client_id])
            client2loaders[client_id] = train_dl_local

        global_train_dl, test_dl = get_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                                    train_bs = args.batch_size, test_bs = args.batch_size)
    

    # Random client sampling support
    clients_per_round = int(args.nclients * args.fraction)
    client_ls = [i for i in range(args.nclients)]
    client_ls_rounds = []
    if clients_per_round != args.nclients:
        for i in range(args.nrounds):
            client_ls_rounds.append(random.sample(client_ls, clients_per_round))
    else:
        for i in range(args.nrounds):
            client_ls_rounds.append(client_ls)
    #=============== Dataset setup =================

    #=============== Model setup ===================
    logger.info("Initializing models")
    client2nets = init_model(args.nclients, args)
    global_net = init_model(1, args)[0]
    #=============== Model setup ===================

    #============== Training setup =================
    for comm_round in range(args.nrounds):
        logger.info("Communication round %d" % comm_round)

        # local training on all selected clients
        nets_current = fedavg_local(args, global_net, logger, client2nets, client2loaders, client_ls_rounds, comm_round, test_dl)

        # global aggregation
        fedavg_global(global_net, client2loaders, nets_current)

        # compute ACC/ASR/train_asr
        global_net.cuda()
        train_acc, train_loss = compute_accuracy(global_net, global_train_dl)
        if args.adv_type == 'CDLS':
            test_acc, asr, train_asr = evaluate_acc_asr(global_net, test_dl, backdoor_test_dl, train_poison_dl)
            global_net.to('cpu')

            logger.info('>> Global Model Train Acc: %f' % train_acc)
            logger.info('>> Global Model Test ACC: %f' % test_acc)
            logger.info('>> Global Model Test ASR: %f' % asr)
            logger.info('>> Global Model Train-Poison ASR: %f' % train_asr)
            logger.info('>> Global Model Train Loss: %f' % train_loss)

            if (comm_round + 1) % args.print_interval == 0:
                print('round: ', str(comm_round))
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Test ACC: %f' % test_acc)
                print('>> Global Model Test ASR: %f' % asr)
                print('>> Global Model Train-Poison ASR: %f' % train_asr)
                print('>> Global Model Train loss: %f' % train_loss)
        elif args.adv_type == 'None':
            test_acc, test_loss = compute_accuracy(global_net, test_dl)
            global_net.to('cpu')

            logger.info('>> Global Model Train Acc: %f' % train_acc)
            logger.info('>> Global Model Test Acc: %f' % test_acc)
            logger.info('>> Global Model Train Loss: %f' % train_loss)

            if (comm_round + 1) % args.print_interval == 0:
                print('round: ', str(comm_round))
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Test accuracy: %f' % test_acc)
                print('>> Global Model Train loss: %f' % train_loss)

        if (comm_round + 1) % args.save_interval == 0:
            torch.save(global_net.state_dict(),
                os.path.join(args.ckptdir, args.aggregation, 'globalmodel_'+args.log_file_name+'.pth'))
            torch.save(client2nets[0].state_dict(),
                os.path.join(args.ckptdir, args.aggregation, 'localmodel0_'+args.log_file_name+'.pth'))
