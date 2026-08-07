import os
import random
import logging
import datetime
import json
import argparse
import torch
from utils import *
from data_utils import *
from models import ResNet18, ResNet34, ResNet50, ResNet18Small, ResNet34Small, ResNet50Small, MobileNetV2, MobileNetV2Large, FangCNN
from attacks.cdls import CDLS_CONFIG, build_cdls_backdoor, pretrain_simclr_digits, pretrain_simclr_domain, train_adv_classifier, train_adv_classifier_domain, AdversaryExtractor, apply_model_poison_constraint
from defenses.graid import graid_aggregate
from data_aug_utils import AutoAugment
from aggregations import fedavg_local, fedavg_global, flame, krum, ndc, deepsight, foolsgold, bnguard


def args_parser():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--dataset", help="dataset", default='digits', type=str,
                        choices=['digits', 'office', 'domain', 'cifar10', 'cifar100'])
    parser.add_argument("--model", help="training model", default="resnet34", type=str,
                        choices=['cnn','resnet18', 'resnet34', 'resnet50', 'mobilenetv2'])
    parser.add_argument("--lr", help="learning rate", default=5e-4, type=float)
    parser.add_argument("--momentum", help="SGD momentum", default=0.9, type=float)
    parser.add_argument("--wd", help="weight decay", default=1e-5, type=float)
    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument('--device', help="cpu, cuda", default="cuda", type=str)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)

    # FL
    parser.add_argument("--aggregation", help="aggregation rule", default='graid', type=str,
                        choices=['fedavg', 'krum', 'flame', 'ndc', 'graid', 'deepsight', 'foolsgold', 'bnguard'])
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

    parser.add_argument('--krum_m', help="number of clients to select for Krum aggregation", default=1, type=int)

    # Defense: GRAID (gradient-inversion reconstruction-based anomaly identification)
    parser.add_argument("--def_num_recon", help="GRAID: # dummy samples reconstructed per client", default=32, type=int)
    parser.add_argument("--def_recon_iters", help="GRAID: gradient-inversion optimization steps per client per round", default=100, type=int)
    parser.add_argument("--def_recon_lr", help="GRAID: Adam learning rate for reconstructing dummy (x, y)", default=0.1, type=float)
    parser.add_argument("--def_tv_weight", help="GRAID: total-variation image-prior weight during reconstruction", default=1e-2, type=float)
    parser.add_argument("--def_recon_every", help="GRAID: run reconstruction+filtering every K rounds (1=every round); the other rounds fall back to plain FedAvg over all clients to save compute", default=3, type=int)
    parser.add_argument("--def_warmup", help="GRAID: # initial warm-up rounds during which GRAID does NOT screen (all clients are FedAvg-aggregated); 0 = no warm-up, GRAID active from round 0", default=0, type=int)
    parser.add_argument("--def_min_cluster", help="GRAID: min reconstructed samples of a class needed to attempt a within-class split", default=6, type=int)
    parser.add_argument("--def_susp_threshold", help="GRAID: discard a client if this fraction of its reconstructions is suspicious", default=0.3, type=float)

    # Adversarial
    parser.add_argument("--adv_type", help="adv type", default='None', type=str,
                        choices=['None', 'CDLS'])
    parser.add_argument("--nbyz", help="# byzantines / # adversarial clients", default=4, type=int)
    parser.add_argument("--bd_target_label", help="original label targeted by the CDLS backdoor", default=0, type=int)
    parser.add_argument("--bd_partition", help="fraction of a client's target_label samples to replace with the nearest cross-domain donor sample", default=0.5, type=float)
    parser.add_argument("--bd_domain", help="digits sub-dataset the clients are assigned to", default='mnist', type=str,
                        choices=['mnist', 'mnist_m', 'svhn', 'syn', 'usps'])
    parser.add_argument("--bd_donor_domains", help="digits sub-datasets donor replacement samples are drawn from; defaults to all domains other than --bd_domain", default=None, type=str, nargs='+')
    parser.add_argument("--bd_donor_pool_size", help="max donor samples per domain kept for the nearest-neighbor search", default=1000, type=int)
    parser.add_argument("--bd_max_search", help="max donor pool entries scanned per victim sample when finding the nearest match", default=500, type=int)

    # CDLS donor-selection distance space (raw pixels vs learned SimCLR features)
    parser.add_argument("--bd_distance", help="donor-selection distance for CDLS", default='pred_kl', type=str,
                        choices=['raw_kl', 'embed_kl', 'pred_kl'])
    parser.add_argument("--bd_simclr_epochs", help="adversary SimCLR pretraining epochs (embed_kl/pred_kl)", default=50, type=int)
    parser.add_argument("--bd_simclr_dim", help="adversary SimCLR projection dim", default=128, type=int)
    parser.add_argument("--bd_simclr_bs", help="adversary SimCLR batch size", default=128, type=int)
    parser.add_argument("--bd_simclr_temp", help="adversary SimCLR NT-Xent temperature", default=0.5, type=float)
    parser.add_argument("--bd_simclr_img_size", help="adversary SimCLR input resolution for DomainNet (natural-image) pretraining; digits/cifar10 always use the 32x32 digit SimCLR and ignore this", default=96, type=int)

    # CDLS evaluation / model-poisoning
    parser.add_argument("--bd_clean_baseline", help="train a clean model but still build the CDLS backdoor test set, to report baseline ASR", action='store_true')
    parser.add_argument("--bd_model_poison", help="enable model-poisoning on top of CDLS data poisoning (stealth reg + constrain-and-scale)", action='store_true')
    parser.add_argument("--bd_stealth_lambda", help="weight of the ||w - w_global||^2 stealth/anomaly-evasion regularizer on malicious clients", default=1e-3, type=float)
    parser.add_argument("--bd_scale", help="malicious update scaling factor for constrain-and-scale (capped at the benign median update norm)", default=2.0, type=float)

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

    adv_clients = []  # the single malicious-client list (data + model poisoning); empty for clean / no-attack runs

    if args.adv_type == 'CDLS':
        if args.dataset not in CDLS_CONFIG:
            raise NotImplementedError(
                "CDLS is implemented for dataset in %s; got '%s'" % (list(CDLS_CONFIG.keys()), args.dataset))
        cfg = CDLS_CONFIG[args.dataset]
        adv_clients = list(range(args.nbyz))

        # victim domain: digits keeps --bd_domain; domain/cifar10 use their fixed default
        victim_domain = args.bd_domain if args.dataset == 'digits' else cfg['victim_domain']

        # donor domains: explicit override, else per-dataset defaults (for digits,
        # exclude whichever domain the clients hold so donors stay cross-domain)
        if args.bd_donor_domains is not None:
            donor_domains = args.bd_donor_domains
        elif args.dataset == 'digits':
            donor_domains = [d for d in ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'] if d != victim_domain]
        else:
            donor_domains = cfg['donor_domains']

        # --- adversary-side SimCLR feature extractor (embed_kl / pred_kl only) ---
        extractor = None
        if args.bd_distance != 'raw_kl':
            if args.dataset in ('digits', 'cifar10'):
                # digits & cifar10 share the SAME 32x32 SimCLR: pretrained on the five
                # digit domains, which are also cifar10's OOD donor pool.
                simclr_domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
                logger.info("Adversary pretraining SimCLR (digits/32x32) on %s (distance=%s)" % (simclr_domains, args.bd_distance))
                print("Adversary pretraining SimCLR (digits/32x32, distance=%s) ..." % args.bd_distance)
                encoder = pretrain_simclr_digits(args, simclr_domains, logger=logger)
                classifier = train_adv_classifier(encoder, args, simclr_domains, logger=logger) \
                    if args.bd_distance == 'pred_kl' else None
                extractor = AdversaryExtractor(encoder, classifier)
            elif args.dataset == 'domain':
                # DomainNet gets its own natural-image SimCLR (ResNet18 backbone),
                # pretrained on all DomainNet domains at bd_simclr_img_size.
                sz = args.bd_simclr_img_size
                simclr_domains = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
                logger.info("Adversary pretraining SimCLR (DomainNet/%dx%d) on %s (distance=%s)" % (sz, sz, simclr_domains, args.bd_distance))
                print("Adversary pretraining SimCLR (DomainNet/%dx%d, distance=%s) ..." % (sz, sz, args.bd_distance))
                encoder = pretrain_simclr_domain(args, simclr_domains, img_size=sz, logger=logger)
                classifier = train_adv_classifier_domain(encoder, args, simclr_domains, img_size=sz, logger=logger) \
                    if args.bd_distance == 'pred_kl' else None
                extractor = AdversaryExtractor(
                    encoder, classifier, img_size=sz,
                    normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
            else:
                raise NotImplementedError(
                    "embed_kl / pred_kl are not supported for dataset='%s'" % args.dataset)

        # --- clean-model baseline: build the SAME backdoor test set but poison NO
        #     client (data or model), so the reported ASR is the natural
        #     false-trigger rate of a clean model (an honest lower bound) ---
        if args.bd_clean_baseline:
            logger.info("CLEAN BASELINE: training a clean model, reporting baseline ASR on the CDLS test set")
            print("CLEAN BASELINE: no training client is poisoned; reporting baseline ASR")
            adv_clients = []  # emptied -> no data poisoning and no model poisoning; test set still built below

        logger.info("Building CDLS backdoor (dataset=%s, victim=%s, donors=%s, target_label=%s, partition=%s, adv_clients=%s, distance=%s)"
                    % (args.dataset, str(victim_domain), str(donor_domains), str(args.bd_target_label),
                       str(args.bd_partition), str(adv_clients), args.bd_distance))

        # Build poisoned train loaders for the malicious clients + adversary test set
        client2loaders, test_dl, backdoor_test_dl, train_poison_dl = build_cdls_backdoor(
            args, client2dataidx, adv_clients, args.bd_target_label, args.bd_partition,
            dataset=args.dataset, domain=victim_domain, donor_domains=donor_domains,
            donor_pool_size=args.bd_donor_pool_size, max_search=args.bd_max_search, seed=args.init_seed,
            distance_mode=args.bd_distance, extractor=extractor)

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

        # local training on all selected clients (malicious ones may add a stealth
        # regularizer when --bd_model_poison is set)
        nets_current = fedavg_local(args, global_net, logger, client2nets, client2loaders,
                                    client_ls_rounds, comm_round, test_dl, adv_clients=adv_clients)

        # optional model-poisoning: constrain-and-scale the malicious updates so
        # they survive robust aggregation (Multi-Krum / FLAME)
        if args.adv_type == 'CDLS' and args.bd_model_poison and adv_clients:
            round_adv = [c for c in adv_clients if c in nets_current]
            if round_adv:
                apply_model_poison_constraint(global_net, nets_current, round_adv, scale=args.bd_scale)

        # global aggregation (dispatch on the chosen rule)
        if args.aggregation == 'krum':
            krum(global_net, client2loaders, nets_current, nbyz=args.nbyz, m=args.krum_m)
        elif args.aggregation == 'flame':
            flame(global_net, client2loaders, nets_current)
        elif args.aggregation == 'ndc':
            ndc(global_net, client2loaders, nets_current)
        elif args.aggregation == 'graid':
            graid_aggregate(args, global_net, nets_current, client2loaders, comm_round, logger)
        elif args.aggregation == 'foolsgold':
            foolsgold(global_net, client2loaders, nets_current)
        elif args.aggregation == 'deepsight':
            deepsight(global_net, client2loaders, nets_current)
        elif args.aggregation == 'bnguard':
            bnguard(global_net, client2loaders, nets_current)
        else:
            fedavg_global(global_net, client2loaders, nets_current)

        # compute ACC/ASR/train_asr
        global_net.cuda()
        train_acc, train_loss = compute_accuracy(global_net, global_train_dl)

        if args.adv_type == 'CDLS':
            test_acc, asr, train_asr = evaluate_acc_asr(global_net, test_dl, backdoor_test_dl, train_poison_dl)
            global_net.to('cpu')

            asr_tag = 'Baseline ASR' if args.bd_clean_baseline else 'Test ASR'
            logger.info('>> Global Model Train Acc: %f' % train_acc)
            logger.info('>> Global Model Test ACC: %f' % test_acc)
            logger.info('>> Global Model %s: %f' % (asr_tag, asr))
            logger.info('>> Global Model Train-Poison ASR: %f' % train_asr)
            logger.info('>> Global Model Train Loss: %f' % train_loss)

            if (comm_round + 1) % args.print_interval == 0:
                print('round: ', str(comm_round))
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Test ACC: %f' % test_acc)
                print('>> Global Model %s: %f' % (asr_tag, asr))
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
