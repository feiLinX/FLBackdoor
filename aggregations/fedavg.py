import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_accuracy
from attacks.pgd import pgd_project_
from attacks.chameleon import supcon_loss
from attacks.soda import soda_self_reference


def fedavg_global(global_net, client2loaders, nets_this_round):
    total_data_points = sum([len(client2loaders[r].dataset) for r in nets_this_round])
    fed_avg_freqs = [len(client2loaders[r].dataset) / total_data_points for r in nets_this_round]

    w_global = global_net.state_dict()
    for net_id, net in enumerate(nets_this_round.values()):
        net_para = net.state_dict()
        if net_id == 0:
            for key in net_para:
                w_global[key] = net_para[key] * fed_avg_freqs[net_id]
        else:
            for key in net_para:
                w_global[key] += net_para[key] * fed_avg_freqs[net_id]
    global_net.load_state_dict(w_global)


def fedavg_local(args, global_net, logger, client2nets, client2loaders, client_ls_rounds, comm_round, test_dl, adv_clients=None, neuro_mask=None, clean_loaders=None):
    # local training on all selected clients
    client_ls_current = client_ls_rounds[comm_round]
    nets_current = {k: client2nets[k] for k in client_ls_current}

    adv_clients = set(adv_clients) if adv_clients is not None else set()
    model_poison = getattr(args, 'bd_model_poison', False)
    attack = getattr(args, 'adv_type', 'None')

    # distribute the global model
    w_global = global_net.state_dict()
    for net in nets_current.values():
        net.load_state_dict(w_global)

    global_vec = None
    if attack == 'PGD':
        global_vec = torch.nn.utils.parameters_to_vector([p.detach() for p in global_net.parameters()]).detach().cuda()
    
    for client_idx in nets_current:
        net = client2nets[client_idx]
        net.train()
        net.cuda()

        is_mal = client_idx in adv_clients
        cdls_stealth = is_mal and model_poison and attack == 'CDLS' and args.bd_stealth_lambda > 0
        pgd = is_mal and attack == 'PGD'
        neuro = is_mal and attack == 'Neurotoxin' and neuro_mask is not None
        chameleon = is_mal and attack == 'Chameleon'
        soda = is_mal and attack == 'SoDa' and clean_loaders is not None and client_idx in clean_loaders
        ref_params = [p.detach().clone() for p in net.parameters()] if cdls_stealth else None
        mask_cuda = [m.cuda() for m in neuro_mask] if neuro else None

        soda_ref_vec = soda_self_reference(global_net, clean_loaders[client_idx], args) if soda else None

        train_loader = client2loaders[client_idx]
        test_loader = test_dl

        logger.info('Training network %s' % str(client_idx))
        logger.info('n_training: %d' % len(train_loader))
        logger.info('n_test: %d' % len(test_loader))

        train_acc, train_loss = compute_accuracy(net, train_loader)
        test_acc, test_loss = compute_accuracy(net, test_loader)

        logger.info('Before Training: Train acc/loss: %.3f/%.3f | Test acc/loss: %.3f/%.3f' % (train_acc, train_loss, test_acc, test_loss))

        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            loss_ls = []
            for batch_idx, (x, target) in enumerate(train_loader):
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                out, features = net(x, return_features=True)
                loss = criterion(out, target)

                if cdls_stealth:
                    reg = sum(((p - r) ** 2).sum() for p, r in zip(net.parameters(), ref_params))
                    loss = loss + args.bd_stealth_lambda * reg

                if chameleon:
                    loss = loss + args.chameleon_lambda * supcon_loss(features, target, args.chameleon_temp)

                if soda:
                    cur_vec = torch.nn.utils.parameters_to_vector(net.parameters())
                    l2 = torch.norm(cur_vec - soda_ref_vec)
                    cos = torch.nn.functional.cosine_similarity(cur_vec, soda_ref_vec, dim=0)
                    loss = loss + args.soda_l2 * l2 + args.soda_cos * (1 - cos)

                loss.backward()

                if neuro:
                    for p, m in zip(net.parameters(), mask_cuda):
                        if p.grad is not None:
                            p.grad.mul_(m)
                optimizer.step()

                if pgd:
                    pgd_project_(net, global_vec, args.pgd_eps)

                loss_ls.append(loss.item())
            
            epoch_loss = sum(loss_ls) / len(loss_ls)

            logger.info('Epoch %d | Loss: %f' % (epoch, epoch_loss))

        train_acc, train_loss = compute_accuracy(net, train_loader)
        test_acc, test_loss = compute_accuracy(net, test_loader)
        logger.info('After Training: Train acc/loss: %.3f/%.3f | Test acc/loss: %.3f/%.3f' % (train_acc, train_loss, test_acc, test_loss))

        net.to('cpu')  # Move the model back to CPU after training
    
    return nets_current