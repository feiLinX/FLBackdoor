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

    parser.add_argument('--auto_aug', action='store_true', help='whether to apply auto augmentation')
    parser.add_argument('--aug_mult', help="replicate each client's assigned sample indices", default=10, type=int)

    parser.add_argument('--krum_m', help="number of clients to select for Krum aggregation", default=15, type=int)

    # Adversarial
    parser.add_argument("--adv_type", help="adv type", default='None', type=str,
                        choices=['None', 'CDLS', 'PGD', 'Neurotoxin', 'Vanilla', 'Chameleon', 'SoDa'])
    parser.add_argument("--nbyz", help="# byzantines / # adversarial clients", default=2, type=int)
    parser.add_argument("--bd_target_label", help="label targeted by the CDLS backdoor", default=0, type=int)
    parser.add_argument("--bd_partition", help="fraction of a client's target_label samples to replace with the cross-domain sample", default=0.3, type=float)
    parser.add_argument("--bd_domain", help="digits sub-dataset the clients are assigned to", default='mnist', type=str,
                        choices=['mnist', 'mnist_m', 'svhn', 'syn', 'usps'])
    parser.add_argument("--bd_donor_domains", help="digits sub-datasets donor replacement samples are drawn from; defaults to all domains other than --bd_domain", default=None, type=str, nargs='+')
    parser.add_argument("--bd_donor_pool_size", help="max donor samples per domain", default=1000, type=int)
    parser.add_argument("--bd_max_search", help="max donor pool entries scanned per victim sample", default=500, type=int)

    # CDLS
    parser.add_argument("--bd_distance", help="feature extractor for CDLS, embed_kl stands for Encoder, pred_kl stands for Encoder + LP", default='pred_kl', type=str,
                        choices=['raw_kl', 'embed_kl', 'pred_kl', 'random'])
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
    parser.add_argument("--def_recon_every", help="run every K rounds (1=every round)", default=3, type=int)
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
                        default=os.path.normpath(os.path.join(_PROJECT_DIR, os.pardir, os.pardir, 'data')) + os.sep)

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
    client2clean_loaders = None  # SoDa: malicious clients' un-poisoned loaders (self-reference stage)

    if args.adv_type == 'CDLS':
        if args.dataset not in CDLS_CONFIG:
            raise NotImplementedError(
                "CDLS is implemented for dataset in %s; got '%s'" % (list(CDLS_CONFIG.keys()), args.dataset))
        cfg = CDLS_CONFIG[args.dataset]
        adv_clients = list(range(args.nbyz))

        victim_domain = args.bd_domain if args.dataset == 'digits' else cfg['victim_domain']

        if args.bd_donor_domains is not None:
            donor_domains = args.bd_donor_domains
        elif args.dataset == 'digits':
            donor_domains = [d for d in ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'] if d != victim_domain]
        else:
            donor_domains = cfg['donor_domains']

        # --- adversary-side SimCLR feature extractor (embed_kl / pred_kl only) ---
        extractor = None
        if args.bd_distance not in ('raw_kl', 'random'):
            if args.dataset in ('digits', 'cifar10'):
                simclr_domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
                logger.info("Adversary pretraining SimCLR (digits/32x32) on %s (distance=%s)" % (simclr_domains, args.bd_distance))
                print("Adversary pretraining SimCLR (digits/32x32, distance=%s) ..." % args.bd_distance)
                encoder = pretrain_simclr_digits(args, simclr_domains, logger=logger)
                classifier = train_adv_classifier(encoder, args, simclr_domains, logger=logger) \
                    if args.bd_distance == 'pred_kl' else None
                extractor = AdversaryExtractor(encoder, classifier)
            elif args.dataset == 'domain':
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

        logger.info("Building CDLS backdoor (dataset=%s, victim=%s, donors=%s, target_label=%s, partition=%s, adv_clients=%s, distance=%s)"
                    % (args.dataset, str(victim_domain), str(donor_domains), str(args.bd_target_label),
                       str(args.bd_partition), str(adv_clients), args.bd_distance))

        # Build poisoned train loaders for the malicious clients + adversary test set
        client2loaders, test_dl, backdoor_test_dl, local_poison_dl = build_cdls_backdoor(
            args, client2dataidx, adv_clients, args.bd_target_label, args.bd_partition,
            dataset=args.dataset, domain=victim_domain, donor_domains=donor_domains,
            donor_pool_size=args.bd_donor_pool_size, max_search=args.bd_max_search, seed=args.init_seed,
            distance_mode=args.bd_distance, extractor=extractor)

        global_train_dl, _ = get_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                         train_bs=args.batch_size, test_bs=args.batch_size)
        
    elif args.adv_type in ('PGD', 'Neurotoxin', 'Vanilla', 'Chameleon'):
            
            adv_clients = list(range(args.nbyz))
    
            if args.dataset == 'digits':
                victim_domain = args.bd_domain
            elif args.dataset == 'domain':
                victim_domain = 'clipart'
            else:
                victim_domain = None
    
            logger.info("Building %s trigger backdoor (dataset=%s, victim=%s, target_label=%s, poison_frac=%s, trigger_size=%d, adv_clients=%s)"
                        % (args.adv_type, args.dataset, str(victim_domain), str(args.bd_target_label),
                           str(args.bd_partition), args.bd_trigger_size, str(adv_clients)))
    
            client2loaders, test_dl, backdoor_test_dl, local_poison_dl = build_trigger_backdoor(
                args, client2dataidx, adv_clients, args.bd_target_label, args.bd_partition,
                dataset=args.dataset, domain=victim_domain, trigger_size=args.bd_trigger_size,
                trigger_value=args.bd_trigger_value, seed=args.init_seed)
    
            global_train_dl, _ = get_dataloader(args, dataset=args.dataset, data_dir=args.data_dir,
                                             train_bs=args.batch_size, test_bs=args.batch_size)
            
    elif args.adv_type == 'SoDa':

        adv_clients = list(range(args.nbyz))
        victim_domain = args.bd_domain if args.dataset == 'digits' else None

        # OOD source: digits <- cifar10, cifar10 <- digits(mnist); overridable via --soda_ood
        if args.soda_ood is not None:
            ood_dataset = args.soda_ood
        else:
            ood_dataset = 'cifar10' if args.dataset != 'cifar10' else 'digits'
        ood_domain = 'mnist' if ood_dataset == 'digits' else None

        logger.info("Building SoDa OOD backdoor (dataset=%s, victim=%s, ood=%s, target_label=%s, poison_frac=%s, adv_clients=%s)"
                    % (args.dataset, str(victim_domain), ood_dataset, str(args.bd_target_label),
                       str(args.bd_partition), str(adv_clients)))

        client2loaders, test_dl, backdoor_test_dl, local_poison_dl, client2clean_loaders = build_soda_backdoor(
            args, client2dataidx, adv_clients, args.bd_target_label, args.bd_partition,
            dataset=args.dataset, domain=victim_domain, ood_dataset=ood_dataset, ood_domain=ood_domain,
            seed=args.init_seed)

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
