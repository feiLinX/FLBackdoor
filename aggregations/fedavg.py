import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import compute_accuracy


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


def fedavg_local(args, global_net, logger, client2nets, client2loaders, client_ls_rounds, comm_round, test_dl):
    # local training on all selected clients
    client_ls_current = client_ls_rounds[comm_round]
    nets_current = {k: client2nets[k] for k in client_ls_current}

    # distribute the global model
    w_global = global_net.state_dict()
    for net in nets_current.values():
        net.load_state_dict(w_global)
    
    for client_idx in nets_current:
        net = client2nets[client_idx]
        net.train()
        net.cuda()

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
        # Extra loss added here

        for epoch in range(args.epochs):
            loss_ls = []
            for batch_idx, (x, target) in enumerate(train_loader):
                x, target = x.cuda(), target.cuda()
                optimizer.zero_grad()
                out, features = net(x, return_features=True)
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()
                loss_ls.append(loss.item())
            
            epoch_loss = sum(loss_ls) / len(loss_ls)

            logger.info('Epoch %d | Loss: %f' % (epoch, epoch_loss))

        train_acc, train_loss = compute_accuracy(net, train_loader)
        test_acc, test_loss = compute_accuracy(net, test_loader)
        logger.info('After Training: Train acc/loss: %.3f/%.3f | Test acc/loss: %.3f/%.3f' % (train_acc, train_loss, test_acc, test_loss))

        net.to('cpu')  # Move the model back to CPU after training
    
    return nets_current