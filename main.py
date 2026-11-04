import os
import random
import logging
import datetime
import json
import argparse
import itertools
import numpy as np
import torch
from utils import *
from data_utils import *
from models import ResNet18, ResNet34, ResNet50, ResNet18Small, ResNet34Small, ResNet50Small, MobileNetV2, MobileNetV2Large, FangCNN
from attacks.cdls import CDLS_CONFIG, build_cdls_backdoor, pretrain_simclr_digits, pretrain_simclr_domain, train_adv_classifier, train_adv_classifier_domain, AdversaryExtractor, apply_model_poison_constraint
from attacks.pgd import build_trigger_backdoor
from attacks.neurotoxin import compute_neurotoxin_mask
from attacks.vanilla import apply_model_replacement
from attacks.soda import build_soda_backdoor
from defenses.grad import grad_aggregate
from defenses.indicator import build_indicator_data, inject_indicator, indicator
from data_aug_utils import AutoAugment
from aggregations import fedavg_local, fedavg_global, flame, krum, ndc, deepsight, foolsgold, bnguard


_CODES_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CODES_DIR)  # repository root holding codes/, logs/, saved_models/


def _resolve_default_data_dir():
    candidates = [
        os.path.join(_CODES_DIR, 'data'),
        os.path.normpath(os.path.join(_PROJECT_DIR, os.pardir, os.pardir, 'data')),
    ]
    for cand in candidates:
        if os.path.isdir(cand):
            return cand + os.sep
    return candidates[-1] + os.sep


def args_parser():
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument("--dataset", help="dataset", default='digits', type=str,
                        choices=['digits', 'office', 'domain', 'cifar10', 'cifar100'])
    parser.add_argument("--model", help="training model", default="resnet34", type=str,
                        choices=['cnn','resnet18', 'resnet34', 'resnet50', 'mobilenetv2'])
    parser.add_argument("--lr", help="learning rate", default=5e-4, type=float,
                        choices=[5e-4, 1e-2, 5e-2])
    parser.add_argument("--momentum", help="SGD momentum", default=0.9, type=float)
    parser.add_argument("--wd", help="weight decay", default=1e-5, type=float,
                        choices=[1e-5, 4e-5, 5e-4])
    parser.add_argument("--batch_size", help="batch size", default=64, type=int)
    parser.add_argument('--device', help="cpu, cuda", default="cuda", type=str)
    parser.add_argument("--gpu", help="index of gpu", default=0, type=int)

    # FL
    parser.add_argument("--aggregation", help="aggregation rule/defense", default='fedavg', type=str,
                        choices=['fedavg', 'krum', 'flame', 'ndc', 'grad', 'deepsight', 'foolsgold', 'bnguard', 'indicator'])
    parser.add_argument("--nrounds", help="# global rounds", default=35, type=int)
    parser.add_argument("--epochs", help="# local epochs", default=5, type=int)
    parser.add_argument("--nclients", help="# clients", default=20, type=int)
    parser.add_argument("--fraction", help="fraction of clients", default=1, type=float)
    parser.add_argument("--bias", help="degree of label non-iidness", default=1, type=float)
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--partition', type=str, default='noniid', help='the data partitioning strategy, iid or noniid')
    parser.add_argument("--num_domains", help="# domains the clients' training data is drawn from (each client holds one domain); "
                        "1 = all clients on --bd_domain (digits) / clipart (domain); k in [2, N-1] runs all C(N, k) domain combinations and averages ACC / ASR",
                        default=1, type=int)
    parser.add_argument("--combo_ids", help="run only these indices of the C(N, k) domain combinations (default: all)", default=None, type=int, nargs='+')

    parser.add_argument('--auto_aug', action='store_true', help='whether to apply auto augmentation')
    parser.add_argument('--aug_mult', help="replicate each client's assigned sample indices", default=10, type=int)

    parser.add_argument('--krum_m', help="number of clients to select for Krum aggregation", default=15, type=int)

    # Adversarial
    parser.add_argument("--adv_type", help="adv type", default='None', type=str,
                        choices=['None', 'CDLS', 'PGD', 'Neurotoxin', 'Vanilla', 'Chameleon', 'SoDa'])
    parser.add_argument("--nbyz", help="# byzantines / # adversarial clients", default=2, type=int)
    parser.add_argument("--bd_target_label", help="label targeted by the CDLS backdoor", default=0, type=int)
    parser.add_argument("--bd_partition", help="fraction of a client's target_label samples to replace with the cross-domain sample", default=0.3, type=float)
    parser.add_argument("--bd_domain", help="digits sub-dataset all clients are assigned to when --num_domains 1", default='mnist', type=str,
                        choices=['mnist', 'mnist_m', 'svhn', 'syn', 'usps'])
    parser.add_argument("--bd_donor_domains", help="sub-datasets CDLS donor replacement samples may be drawn from (the clients' own domains are always excluded); defaults to every domain no client trains on", default=None, type=str, nargs='+')
    parser.add_argument("--bd_donor_pool_size", help="max donor samples per domain", default=1000, type=int)
    parser.add_argument("--bd_max_search", help="max donor pool entries scanned per victim sample", default=500, type=int)

    # CDLS
    parser.add_argument("--bd_distance", help="feature distance for CDLS donor selection, <repr>_<metric>: repr raw (pixels) / embed (Encoder) / pred (Encoder + LP), metric kl / euclid / cos", default='pred_kl', type=str,
                        choices=['raw_kl', 'embed_kl', 'pred_kl', 'raw_euclid', 'embed_euclid', 'pred_euclid',
                                 'raw_cos', 'embed_cos', 'pred_cos', 'random'])
    parser.add_argument("--bd_simclr_epochs", help="adversary SimCLR pretraining epochs", default=50, type=int)
    parser.add_argument("--bd_simclr_dim", help="projection dim", default=128, type=int)
    parser.add_argument("--bd_simclr_bs", help="batch size", default=128, type=int)
    parser.add_argument("--bd_simclr_temp", help="temperature", default=0.5, type=float)
    parser.add_argument("--bd_simclr_img_size", help="adversary SimCLR input resolution for DomainNet", default=96, type=int)

    parser.add_argument("--bd_model_poison", help="enable model-poisoning (stealth reg + constrain-and-scale)", action='store_true')
    parser.add_argument("--bd_stealth_lambda", help="weight of the regularizer on malicious clients", default=1e-3, type=float)
    parser.add_argument("--bd_scale", help="scaling factor for constrain-and-scale", default=2.0, type=float)

    # GRAD
    parser.add_argument("--def_num_recon", help="# dummy samples reconstructed per client", default=48, type=int)
    parser.add_argument("--def_recon_iters", help="gradient-inversion optimization steps", default=100, type=int)
    parser.add_argument("--def_recon_lr", help="Adam learning rate for reconstructing dummy x and y", default=0.1, type=float)
    parser.add_argument("--def_tv_weight", help="total-variation image-prior weight", default=1e-2, type=float)
    parser.add_argument("--def_recon_every", help="run every K rounds (1=every round)", default=1, type=int)
    parser.add_argument("--def_warmup", help="# initial warm-up rounds", default=0, type=int)
    parser.add_argument("--def_min_cluster", help="min reconstructed samples of a class needed to attempt a split", default=6, type=int)
    parser.add_argument("--def_sep_ratio", help="threshold of accepting the KMeans 2-way split", default=3.0, type=float)
    parser.add_argument("--def_susp_threshold", help="discard a client if this fraction of reconstructions is suspicious", default=0.51, type=float)

    # Indicator
    parser.add_argument("--ind_ood", help="OOD dataset the server's watermark probes", default=None, type=str, choices=['cifar10', 'digits'])
    parser.add_argument("--ind_samples", help="# fixed OOD probe images", default=100, type=int)
    parser.add_argument("--ind_inject_epochs", help="# epochs the server fine-tunes the global model", default=10, type=int)
    parser.add_argument("--ind_inject_lr", help="SGD learning rate", default=0.01, type=float)
    parser.add_argument("--ind_mu", help="weight of the proximal term during watermark injection", default=0.0, type=float)
    parser.add_argument("--ind_threshold", help="discard a client whose watermark accuracy stays >= ", default=0.5, type=float)

    # PGD, Neurotoxin 
    parser.add_argument("--bd_trigger_size", help="side length of the square corner trigger", default=5, type=int)
    parser.add_argument("--bd_trigger_value", help="pixel value stamped for the trigger", default=255, type=int)
    parser.add_argument("--pgd_eps", help="L2 radius of the ball ", default=1.0, type=float)
    parser.add_argument("--neuro_mask_ratio", help="fraction of top benign-gradient coordinates malicious clients freeze", default=0.1, type=float)

    # Vanilla, Chameleon, SoDa 
    parser.add_argument("--chameleon_lambda", help="weight of the supervised-contrastive loss on malicious clients", default=1.0, type=float)
    parser.add_argument("--chameleon_temp", help="temperature of the supervised-contrastive loss", default=0.5, type=float)
    parser.add_argument("--soda_ood", help="OOD dataset the backdoor images are drawn from", default=None, type=str, choices=['cifar10', 'digits'])
    parser.add_argument("--soda_l2", help="weight of the l_2 term on malicious clients", default=0.1, type=float)
    parser.add_argument("--soda_cos", help="weight of the cos similarity term on malicious clients", default=100.0, type=float)

    # Logging
    parser.add_argument("--data_dir", type=str, required=False,
                        default=_resolve_default_data_dir())

    parser.add_argument('--logdir', type=str, required=False,
                        default=os.path.join(_PROJECT_DIR, 'logs') + os.sep)

    parser.add_argument('--log_file_name', type=str, default=None, help='The log file name')
    parser.add_argument('--ckptdir', type=str, required=False,
                        default=os.path.join(_PROJECT_DIR, 'saved_models') + os.sep)
    
    parser.add_argument('--print_interval', type=int, default=5,
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


def get_domain_combos(args):
    """Client-domain combinations to run: --num_domains 1 keeps the single default domain,
    --num_domains k >= 2 enumerates every k-subset of the dataset's N domains (k <= N-1)."""
    if args.num_domains == 1:
        if args.dataset == 'digits':
            combos = [(args.bd_domain,)]
        elif args.dataset == 'domain':
            combos = [('clipart',)]
        else:
            combos = [(None,)]  # dataset's default (only) domain
    else:
        if args.dataset not in DATASET_DOMAINS:
            raise ValueError("--num_domains > 1 needs a dataset in %s; got '%s'" % (list(DATASET_DOMAINS.keys()), args.dataset))
        all_domains = DATASET_DOMAINS[args.dataset]
        if not 2 <= args.num_domains <= len(all_domains) - 1:
            raise ValueError("--num_domains must be in [1, %d] for dataset '%s' (at least one domain is left for CDLS); got %d"
                             % (len(all_domains) - 1, args.dataset, args.num_domains))
        if args.num_domains > args.nclients:
            raise ValueError("--num_domains (%d) cannot exceed --nclients (%d)" % (args.num_domains, args.nclients))
        combos = list(itertools.combinations(all_domains, args.num_domains))

    if args.combo_ids is not None:
        combos = [combos[i] for i in args.combo_ids]

    return combos


def build_extractor(args, logger):
    """Adversary-side SimCLR feature extractor for CDLS embed_* / pred_* (None otherwise); it does not depend on
    the clients' domains, so it is trained once and shared by all domain combinations."""
    if args.adv_type != 'CDLS' or not args.bd_distance.startswith(('embed_', 'pred_')):
        return None

    if args.dataset in ('digits', 'cifar10'):
        simclr_domains = DATASET_DOMAINS['digits']
        logger.info("Adversary pretraining SimCLR (digits/32x32) on %s (distance=%s)" % (simclr_domains, args.bd_distance))
        print("Adversary pretraining SimCLR (digits/32x32, distance=%s) ..." % args.bd_distance)
        encoder = pretrain_simclr_digits(args, simclr_domains, logger=logger)
        classifier = train_adv_classifier(encoder, args, simclr_domains, logger=logger) \
            if args.bd_distance.startswith('pred_') else None
        extractor = AdversaryExtractor(encoder, classifier)
    elif args.dataset == 'domain':
        sz = args.bd_simclr_img_size
        simclr_domains = DATASET_DOMAINS['domain']
        logger.info("Adversary pretraining SimCLR (DomainNet/%dx%d) on %s (distance=%s)" % (sz, sz, simclr_domains, args.bd_distance))
        print("Adversary pretraining SimCLR (DomainNet/%dx%d, distance=%s) ..." % (sz, sz, args.bd_distance))
        encoder = pretrain_simclr_domain(args, simclr_domains, img_size=sz, logger=logger)
        classifier = train_adv_classifier_domain(encoder, args, simclr_domains, img_size=sz, logger=logger) \
            if args.bd_distance.startswith('pred_') else None
        extractor = AdversaryExtractor(
            encoder, classifier, img_size=sz,
            normalize=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    else:
        raise NotImplementedError(
            "embed_* / pred_* distances are not supported for dataset='%s'" % args.dataset)

    return extractor


def run_fl(args, client_domains, logger, extractor=None, tag=''):
    """One FL experiment with the clients spread over `client_domains`; returns the per-round ACC / ASR."""
    seed_everything(args.init_seed)
    history = {'acc': [], 'asr': []}

    #=============== Dataset setup ===============
    logger.info("Partitioning data")
    client2dataidx, client2domain = partition_multi_domain(args.dataset, args.data_dir, args.partition,
                                                           args.nclients, args.bias, client_domains)
    logger.info("Client domains: %s" % str(client2domain))

    # union of the client domains: full train set (train acc / Neurotoxin mask) and clean test set
    global_train_dl, test_dl = get_multi_domain_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                                           train_bs=args.batch_size, test_bs=args.batch_size,
                                                           domains=client_domains)

    adv_clients = []  # the single malicious-client list (data + model poisoning); empty for clean / no-attack runs
    client2clean_loaders = None  # SoDa: malicious clients' un-poisoned loaders (self-reference stage)

    if args.adv_type == 'CDLS':
        cfg = CDLS_CONFIG[args.dataset]
        adv_clients = list(range(args.nbyz))

        # CDLS attacks with the domains no client trains on
        donor_domains = DATASET_DOMAINS[cfg['donor_dataset']] if args.bd_donor_domains is None else args.bd_donor_domains
        donor_domains = [d for d in donor_domains if d not in client_domains]
        if not donor_domains:
            raise ValueError("no CDLS donor domain left outside the client domains %s" % str(client_domains))
        history['donor_domains'] = donor_domains

        logger.info("Building CDLS backdoor (dataset=%s, client_domains=%s, donors=%s, target_label=%s, partition=%s, adv_clients=%s, distance=%s)"
                    % (args.dataset, str(client_domains), str(donor_domains), str(args.bd_target_label),
                       str(args.bd_partition), str(adv_clients), args.bd_distance))

        # Build poisoned train loaders for the malicious clients + adversary test set
        client2loaders, test_dl, backdoor_test_dl, local_poison_dl = build_cdls_backdoor(
            args, client2dataidx, client2domain, adv_clients, args.bd_target_label, args.bd_partition,
            dataset=args.dataset, donor_domains=donor_domains,
            donor_pool_size=args.bd_donor_pool_size, max_search=args.bd_max_search, seed=args.init_seed,
            distance_mode=args.bd_distance, extractor=extractor)

    elif args.adv_type in ('PGD', 'Neurotoxin', 'Vanilla', 'Chameleon'):

        adv_clients = list(range(args.nbyz))

        logger.info("Building %s trigger backdoor (dataset=%s, client_domains=%s, target_label=%s, poison_frac=%s, trigger_size=%d, adv_clients=%s)"
                    % (args.adv_type, args.dataset, str(client_domains), str(args.bd_target_label),
                       str(args.bd_partition), args.bd_trigger_size, str(adv_clients)))

        client2loaders, test_dl, backdoor_test_dl, local_poison_dl = build_trigger_backdoor(
            args, client2dataidx, client2domain, adv_clients, args.bd_target_label, args.bd_partition,
            dataset=args.dataset, trigger_size=args.bd_trigger_size,
            trigger_value=args.bd_trigger_value, seed=args.init_seed)

    elif args.adv_type == 'SoDa':

        adv_clients = list(range(args.nbyz))

        # OOD source: digits <- cifar10, cifar10 <- digits(mnist); overridable via --soda_ood
        if args.soda_ood is not None:
            ood_dataset = args.soda_ood
        else:
            ood_dataset = 'cifar10' if args.dataset != 'cifar10' else 'digits'
        ood_domain = 'mnist' if ood_dataset == 'digits' else None

        logger.info("Building SoDa OOD backdoor (dataset=%s, client_domains=%s, ood=%s, target_label=%s, poison_frac=%s, adv_clients=%s)"
                    % (args.dataset, str(client_domains), ood_dataset, str(args.bd_target_label),
                       str(args.bd_partition), str(adv_clients)))

        client2loaders, test_dl, backdoor_test_dl, local_poison_dl, client2clean_loaders = build_soda_backdoor(
            args, client2dataidx, client2domain, adv_clients, args.bd_target_label, args.bd_partition,
            dataset=args.dataset, ood_dataset=ood_dataset, ood_domain=ood_domain,
            seed=args.init_seed)

    elif args.adv_type == 'None':
        logger.info("No attack selected; training on clean data only")

        client2loaders = {}
        for client_id in range(args.nclients):
            train_dl_local, _ = get_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                            train_bs=args.batch_size, test_bs=args.batch_size, dataidxs=client2dataidx[client_id],
                                            domain=client2domain[client_id])
            client2loaders[client_id] = train_dl_local


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

    indicator_loader = None
    if args.aggregation == 'indicator':
        indicator_loader = build_indicator_data(args, ood_dataset=args.ind_ood,
                                                n_samples=args.ind_samples, seed=args.init_seed)

    #============== Training setup =================
    for comm_round in range(args.nrounds):
        logger.info("Communication round %d" % comm_round)

        neuro_mask = None
        if args.adv_type == 'Neurotoxin' and adv_clients:
            neuro_mask = compute_neurotoxin_mask(global_net, global_train_dl, args.neuro_mask_ratio)

        indicator_wm_bn = None
        if args.aggregation == 'indicator':
            indicator_wm_bn = inject_indicator(global_net, indicator_loader,
                                               lr=args.ind_inject_lr, epochs=args.ind_inject_epochs,
                                               momentum=args.momentum, mu=args.ind_mu)

        nets_current = fedavg_local(args, global_net, logger, client2nets, client2loaders,
                                    client_ls_rounds, comm_round, test_dl, adv_clients=adv_clients,
                                    neuro_mask=neuro_mask, clean_loaders=client2clean_loaders)


        if args.bd_model_poison and adv_clients:
            round_adv = [c for c in adv_clients if c in nets_current]
            if round_adv:
                if args.adv_type == 'Vanilla':
                    apply_model_replacement(global_net, nets_current, round_adv, scale=args.bd_scale)
                elif args.adv_type in ('CDLS', 'PGD', 'Neurotoxin', 'Chameleon'):
                    apply_model_poison_constraint(global_net, nets_current, round_adv, scale=args.bd_scale)

        # global aggregation (dispatch on the chosen rule)
        if args.aggregation == 'krum':
            krum(global_net, client2loaders, nets_current, nbyz=args.nbyz, m=args.krum_m)
        elif args.aggregation == 'flame':
            flame(global_net, client2loaders, nets_current)
        elif args.aggregation == 'ndc':
            ndc(global_net, client2loaders, nets_current)
        elif args.aggregation == 'grad':
            grad_aggregate(args, global_net, nets_current, client2loaders, comm_round, logger)
        elif args.aggregation == 'foolsgold':
            foolsgold(global_net, client2loaders, nets_current)
        elif args.aggregation == 'deepsight':
            deepsight(global_net, client2loaders, nets_current)
        elif args.aggregation == 'bnguard':
            bnguard(global_net, client2loaders, nets_current)
        elif args.aggregation == 'indicator':
            indicator(global_net, client2loaders, nets_current, indicator_loader,
                      wm_bn=indicator_wm_bn, threshold=args.ind_threshold, logger=logger)
        else:
            fedavg_global(global_net, client2loaders, nets_current)


        global_net.cuda()
        train_acc, train_loss = compute_accuracy(global_net, global_train_dl)

        if args.adv_type in ('CDLS', 'PGD', 'Neurotoxin', 'Vanilla', 'Chameleon', 'SoDa'):
            test_acc, test_asr, local_test_asr = evaluate_acc_asr(global_net, test_dl, backdoor_test_dl, local_poison_dl)
            global_net.to('cpu')
            global_asr = test_asr if args.dataset == 'digits' else max(test_asr, local_test_asr)
            history['acc'].append(test_acc)
            history['asr'].append(global_asr)
            logger.info('>> Global Model Train Acc: %f' % train_acc)
            logger.info('>> Global Model Test ACC: %f' % test_acc)
            logger.info('>> Global Model ASR: %f' % global_asr)
            logger.info('>> Global Model Train Loss: %f' % train_loss)

            if (comm_round + 1) % args.print_interval == 0:
                print('round: ', str(comm_round))
                print('>> Global Model Train accuracy: %f' % train_acc)
                print('>> Global Model Test ACC: %f' % test_acc)
                print('>> Global Model ASR: %f' % global_asr)
                print('>> Global Model Train loss: %f' % train_loss)

        elif args.adv_type == 'None':
            test_acc, test_loss = compute_accuracy(global_net, test_dl)
            global_net.to('cpu')
            history['acc'].append(test_acc)

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
                os.path.join(args.ckptdir, args.aggregation, 'globalmodel_'+args.log_file_name+tag+'.pth'))
            torch.save(client2nets[0].state_dict(),
                os.path.join(args.ckptdir, args.aggregation, 'localmodel0_'+args.log_file_name+tag+'.pth'))

    return history


def summarize_results(args, results, logger):
    """Report each combination's final-round ACC / ASR and their mean (std) over all combinations; the per-round
    curves of every combination and their mean go to <logdir>/results-<log_file_name>.json."""
    metrics = ['acc'] if args.adv_type == 'None' else ['acc', 'asr']

    lines = ['===== Summary over %d domain combination(s), final round =====' % len(results)]
    for r in results:
        for m in metrics:
            r['final_' + m] = r[m][-1]
        line = '[%s]' % '+'.join(str(d) for d in r['client_domains'])
        if 'donor_domains' in r:
            line += ' (CDLS donors: %s)' % '+'.join(r['donor_domains'])
        lines.append(line + ' ' + ' | '.join('%s: %f' % (m.upper(), r['final_' + m]) for m in metrics))

    summary = {'args': vars(args), 'combos': results}
    for m in metrics:
        finals = np.array([r['final_' + m] for r in results])
        summary['mean_final_' + m] = float(finals.mean())
        summary['std_final_' + m] = float(finals.std())
        summary['mean_%s_per_round' % m] = np.mean([r[m] for r in results], axis=0).tolist()
        lines.append('>> Mean %s over combinations: %f (std %f)' % (m.upper(), finals.mean(), finals.std()))

    for line in lines:
        logger.info(line)
        print(line)

    with open(os.path.join(args.logdir, 'results-%s.json' % args.log_file_name), 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":

    args = args_parser()
    print(args)
    #=============== Logging setup ===============
    mkdirs(args.logdir)
    mkdirs(args.ckptdir)
    mkdirs(os.path.join(args.ckptdir, args.aggregation))

    if args.log_file_name is None:
        argument_path = 'exp_args-%s' % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    else:
        argument_path = 'exp_args-%s' % args.log_file_name

    argument_path = argument_path + '.json'

    with open(os.path.join(args.logdir, argument_path), 'w') as f:
        json.dump(str(args), f)

    if args.log_file_name is None:
        args.log_file_name = 'exp_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S"))

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

    if args.adv_type == 'CDLS' and args.dataset not in CDLS_CONFIG:
        raise NotImplementedError(
            "CDLS is implemented for dataset in %s; got '%s'" % (list(CDLS_CONFIG.keys()), args.dataset))

    combos = get_domain_combos(args)

    seed_everything(args.init_seed)
    extractor = build_extractor(args, logger)

    results = []
    for combo_i, client_domains in enumerate(combos):
        combo_name = '+'.join(str(d) for d in client_domains)
        msg = '===== Domain combination %d/%d: clients on [%s] =====' % (combo_i + 1, len(combos), combo_name)
        logger.info(msg)
        print(msg)

        tag = '' if args.num_domains == 1 else '_' + combo_name  # keep per-combination checkpoints apart
        history = run_fl(args, list(client_domains), logger, extractor=extractor, tag=tag)
        results.append({'client_domains': list(client_domains), **history})

    summarize_results(args, results, logger)
