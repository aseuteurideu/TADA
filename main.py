import argparse
import copy
import random

import torch
from torch import nn
import sys
import numpy as np
import os
from dataset import ASVspoof2019_LA
from torch.utils.data import DataLoader
import pickle
from tensorboardX import SummaryWriter
from utils import save_checkpoint
from model import RawNet
import yaml
from tqdm import tqdm
from utils import our_attack
from model_aasist import AASIST

# some default values taken from https://github.com/asvspoof-challenge/2021/blob/main/LA/Baseline-RawNet2/main.py
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=7, type=int)
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
parser.add_argument('--num_epochs', default=100, type=int, help='number of total epochs to run')
parser.add_argument('--net', default='resnet18', type=str)
parser.add_argument('--final_dim', default=1024, type=int, help='length of vector output from audio/video subnetwork')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--resume', default='', type=str, help='path of model to resume training')
parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')
parser.add_argument('--test', default='', type=str)
parser.add_argument('--seed', type=int, default=1234, help='random seed (default: 1234)')
parser.add_argument('--model_config', default='config/model_config_RawNet.yaml', type=str, help='path of model config')
parser.add_argument('--save_all', action='store_true', help='Save all weights')

parser.add_argument('--fake_type', default='', type=str, help='Fake type for test')

##############################
# adversarial attack as pseudo anomaly
parser.add_argument('--atk_prob', default=0, type=float, help='Probability of changing data to the attacked version')
parser.add_argument('--atk_type', default='half_fake', type=str, choices=['fake','half_fake'] , help='if atk_prob > 0. fake: attack so the data become fake. half_fake: attack so the data become half fake.')
parser.add_argument('--atk_epsmin', default=0.01, type=float, help='attack epsilon minimum range')
parser.add_argument('--atk_epsmax', default=0.5, type=float, help='attack epsilon minimum range')

##################################
# add noise as pseudo anomaly
parser.add_argument('--noise_prob', default=0, type=float, help='Probability of changing data to the noisy')
parser.add_argument('--noise_std_min', default=0.01, type=float, help='Noise std min')
parser.add_argument('--noise_std_max', default=0.01, type=float, help='Noise std max')
parser.add_argument('--noise_mean', default=0, type=float, help='Noise mean')

def cosine_annealing(step, total_steps, lr_max, lr_min):
    """Cosine Annealing for learning rate decay scheduler"""
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def train_epoch(train_loader, model, lr, optim, device, scheduler):
    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)  # bcos way more fake, put more weight on real class
    criterion = nn.CrossEntropyLoss(weight=weight)

    attack_criterion = nn.CrossEntropyLoss()

    if args.atk_prob > 0:
        assert args.noise_prob == 0
    if args.noise_prob > 0:
        assert args.atk_prob == 0

    for it, (batch_x, batch_y) in enumerate(train_loader):

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).to(device)

        # attack
        if args.atk_prob > 0:
            batch_x_atk = copy.deepcopy(batch_x)
            batch_x_atk.requires_grad = True

            batch_out = model(batch_x_atk)

            if args.atk_type == 'half_fake':
                attack_label = torch.ones([batch_x.shape[0], 2]).to(device) * 0.5
            else:
                assert args.atk_type == 'fake'
                attack_label = torch.zeros_like(batch_y)

            attack_loss = attack_criterion(batch_out, attack_label)

            model.zero_grad()
            attack_loss.backward()
            data_grad = batch_x_atk.grad.data

            epsilon = random.uniform(args.atk_epsmin, args.atk_epsmax)
            attacked_x = our_attack(batch_x_atk, epsilon, data_grad)

            for data_idx in range(batch_x.shape[0]):
                to_change = random.uniform(0, 1) < args.atk_prob
                if to_change:
                    batch_x[data_idx] = attacked_x[data_idx].detach()
                    batch_y[data_idx] = 0  # fake

        elif args.noise_prob > 0:
            batch_x_atk = copy.deepcopy(batch_x)
            std, mean = random.uniform(args.noise_std_min, args.noise_std_max), args.noise_mean
            noise = torch.randn_like(batch_x_atk) * std + mean

            attacked_x = batch_x_atk + noise
            attacked_x = torch.clamp(attacked_x, -1, 1)

            for data_idx in range(batch_x.shape[0]):
                to_change = random.uniform(0, 1) < args.noise_prob
                if to_change:
                    batch_x[data_idx] = attacked_x[data_idx].detach()
                    batch_y[data_idx] = 0  # fake


        batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        num_loss = 1

        batch_loss = batch_loss / num_loss
        running_loss += (batch_loss.item() * batch_size)
        if ii % 10 == 0:
            write_text = '\r \t train_acc: {:.2f}'.format((num_correct / num_total) * 100)
            sys.stdout.write(write_text)

        optim.zero_grad()
        batch_loss.backward()
        optim.step()

        if scheduler is not None:
            scheduler.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100

    return running_loss, train_accuracy


def evaluate_accuracy(val_loader, model, device):
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for it, (batch_x, batch_y) in enumerate(val_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

    val_acc = 100 * (num_correct / num_total)

    return val_acc

def test_epoch(test_loader, model, device):
    test_pred = []
    test_target = []
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    for idx, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader)):
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).to(device)
        batch_out = model(batch_x)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()

        test_pred.extend(batch_out.tolist())
        test_target.extend(batch_y.tolist())

    return test_pred, test_target

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = parser.parse_args()

    # make experiment reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    # model naming
    model_tag = 'v1_{}_{}_{}_{}'.format('LA', args.num_epochs, args.batch_size, args.lr)
    if args.atk_prob > 0:
        model_tag += '_atk' + str(args.atk_prob)
        model_tag += '_aty' + args.atk_type if args.atk_type != 'half_fake' else ''
        model_tag += '_ate' + str(args.atk_epsmin) + '-' + str(args.atk_epsmax)
    if args.noise_prob > 0:
        if args.noise_std_min == args.noise_std_max:
            model_tag += '_noi' + str(args.noise_prob) + '-' + str(args.noise_mean) + '-' + str(args.noise_std_min)
        else:
            model_tag += '_noi' + str(args.noise_prob) + '-' + str(args.noise_mean) + '-' + str(args.noise_std_min) + '-' + str(args.noise_std_max)

    model_tag += '_' + os.path.basename(args.model_config)[:-5] if args.model_config != 'config/model_config_RawNet.yaml' else ''
    print(model_tag)
    model_save_path = os.path.join('log_tmp', model_tag)

    # set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    if not os.path.exists(os.path.join(model_save_path, 'model')):
        os.mkdir(os.path.join(model_save_path, 'model'))

    # GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))

    # model
    with open(args.model_config, 'r') as f_yaml:
        parser1 = yaml.safe_load(f_yaml)

    if 'RawNet' in args.model_config:
        model = RawNet(parser1['model'], device)
    else:  # elif 'aasist' in args.model_config:
        if not args.test:
            assert args.num_epochs == 100
            assert args.batch_size == 16  #24
        model = AASIST(parser1['model'])
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = (model).to(device)

    # set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.test:
        assert os.path.isfile(args.test)
        if args.model_config != "config/model_config_RawNet.yaml":
            assert '_' + os.path.basename(args.model_config)[:-5] in args.test
        else:
            assert 'RawNet' not in args.test and 'Transformer' not in args.test and 'ShallowNet' not in args.test and 'RawGAT_ST' not in args.test

        print("=> loading test '{}'".format(args.test))
        checkpoint = torch.load(args.test, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'])

        test_set = ASVspoof2019_LA(split='test', fake_type=args.fake_type)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)
        test_pred, test_target = test_epoch(test_loader, model, device)

        # test result save folder
        paths = args.test.split('/')
        save_folder = os.path.join('test_results', paths[1], paths[-1][:-8])
        if not os.path.exists(save_folder): os.makedirs(save_folder)

        file_pred = open(os.path.join(save_folder, args.fake_type + "file_pred.pkl"), "wb")
        pickle.dump(test_pred, file_pred)
        file_pred.close()
        file_target = open(os.path.join(save_folder, args.fake_type + "file_target.pkl"), "wb")
        pickle.dump(test_target, file_target)
        file_target.close()

        sys.exit()

    model = nn.DataParallel(model)
    if args.resume:
        assert os.path.isfile(args.resume)

        print("=> loading resumed checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        args.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded resumed checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    # define train dataloader
    train_set = ASVspoof2019_LA(split='train', fake_type=args.fake_type)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # define validation dataloader
    val_set = ASVspoof2019_LA(split='val', fake_type=args.fake_type)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    # Scheduler
    if 'aasist' in args.model_config:
        total_steps = args.num_epochs * len(train_loader)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                total_steps,
                1,  # since lr_lambda computes multiplicative factor
                0.000005 / 0.0001))

        if args.resume:
            scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        scheduler = None


    # Training and validation
    num_epochs = args.num_epochs
    writer = SummaryWriter('log_tmp/{}/img'.format(model_tag))
    for epoch in range(args.start_epoch, num_epochs):
        running_loss, train_acc = train_epoch(train_loader, model, args.lr, optimizer, device, scheduler)
        val_acc = evaluate_accuracy(val_loader, model, device)
        writer.add_scalar('train_accuracy', train_acc, epoch)
        writer.add_scalar('valid_accuracy', val_acc, epoch)
        writer.add_scalar('loss', running_loss, epoch)

        print('\n{} - {} - {:.2f} - {:.2f}'.format(epoch,
                                                   running_loss, train_acc, val_acc))

        # save check_point
        if epoch == 0:
            best_acc = val_acc

        is_best = val_acc <= best_acc
        best_acc = max(val_acc, best_acc)

        scheduler_state_dict = scheduler.state_dict() if scheduler is not None else None
        save_checkpoint({
            'epoch': epoch + 1,
            'net': args.net,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler_state_dict
        }, is_best, filename=os.path.join('log_tmp', model_tag, 'model', 'epoch%s.pth.tar' % str(epoch + 1)), keep_all=args.save_all)

