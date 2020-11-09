
import os
import argparse
import math
import copy
import time

import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
import torchvision
from torchvision import transforms

from background_classification.data.config import cfg
from background_classification.model.model import BGClassification
from background_classification.utils.functions import SavePath

import numpy as np
import matplotlib.pyplot as plt


# ================================================
# How to train model
# python train.py --dataset=path/to/dataset
#
# ================================================

def main():
    parse_args()
    # data_show()
    train()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    global args
    parser = argparse.ArgumentParser(
        description='BGClassification Training Script')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Batch size for training')
    parser.add_argument('--start_iter', default=-1, type=int,
                        help='Resume training at this iter. If this is -1, the iteration will be' \
                             'determined from the file name.')
    parser.add_argument('--resume', default=None, type=str,
                        help='Checkpoint state_dict file to resume training from. If this is "interrupt"'\
                         ', the model will resume training from the interrupt file.')
    parser.add_argument('--dataset', required=True, type=str,
                        help='Training Dataset and Validation Dataset directory path')
    parser.add_argument('--cuda', default=True, type=str2bool,
                        help='Use cuda')
    parser.add_argument('--save_directory', default='weights/', type=str,
                        help='Directory for saving checkpoint models.')
    parser.add_argument('--lr', '--learning_rate', default=None, type=float,
                        help='Initial learning rate. Leave as None to read this from the config.')
    parser.add_argument('--momentum', default=None, type=float,
                        help='Momentum for SGD. Leave as None to read this from the config.')
    parser.add_argument('--save_interval', default=1000, type=int,
                        help='The number of iterations between saving the model.')
    parser.add_argument('--gamma', default=None, type=float,
                        help='For each lr step, what to multiply the lr by. Leave as None to read this from the config.')

    parser.set_defaults(keep_latest=False, log=True, log_gpu=False, interrupt=True, autoscale=True)
    args = parser.parse_args()

    # Update training parameters from the config if necessary
    def replace(name):
        if getattr(args, name) == None: setattr(args, name, getattr(cfg, name))

    replace('lr')
    replace('gamma')
    replace('momentum')


def train():
    if not os.path.exists(args.save_directory):
        os.mkdir(args.save_directory)

    data_transform = transforms.Compose([
        transforms.Resize(cfg.max_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    total_dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=data_transform)

    alpha = 0.8
    total_size = len(total_dataset)
    train_size = int(len(total_dataset) * alpha)
    train_dataset, val_dataset = random_split(total_dataset, (train_size, total_size - train_size))
    datasets_size = {'train': train_size,
                     'val': total_size - train_size}
    dataloaders = {'train': DataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=4),
                   'val': DataLoader(val_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4)}

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")

    # Define Model
    net = BGClassification()

    if args.resume == 'interrupt':
        args.resume = SavePath.get_interrupt(args.save_directory)
    elif args.resume == 'latest':
        args.resume = SavePath.get_latest(args.save_directory, cfg.name)

    if args.resume is not None:
        print('Resuming training, loading {}...'.format(args.resume))
        net.load_weights(args.resume)

        if args.start_iter == -1:
            args.start_iter = SavePath.from_str(args.resume).iteration

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    criterion = nn.CrossEntropyLoss()

    if args.cuda:
        net = net.to(device)

    save_path = lambda epoch, iteration: SavePath(cfg.name, epoch, iteration).get_path(root=args.save_directory)

    # epoch calc
    iteration = max(args.start_iter, 0)
    epoch_size = datasets_size['train'] // args.batch_size
    num_epochs = math.ceil(cfg.max_iter / epoch_size)

    best_model_wts = copy.deepcopy(net.state_dict())
    best_acc = 0.0

    since = time.time()

    print("Begin training!")
    try:
        for epoch in range(num_epochs):

            # Resume from start_iter
            if (epoch + 1) * epoch_size < iteration:
                continue

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # 각 에폭(epoch)은 학습 단계와 검증 단계를 갖습니다.
            for phase in ['train', 'val']:
                if phase == 'train':
                    net.train()  # 모델을 학습 모드로 설정
                else:
                    net.eval()  # 모델을 평가 모드로 설정

                running_loss = 0.0
                running_corrects = 0

                st_iter = iteration
                dataloader_size = len(dataloaders[phase])
                # 데이터를 반복
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # 매개변수 경사도를 0으로 설정
                    optimizer.zero_grad()

                    # 순전파
                    # 학습 시에만 연산 기록을 추적
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = net(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 학습 단계인 경우 역전파 + 최적화
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # 통계
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    iteration += 1
                    if iteration % args.save_interval == 0 and iteration != args.start_iter:
                        print('Saving state, iter:', iteration)
                        net.save_weights(save_path(epoch, iteration))

                    percent = int(((iteration - st_iter) / dataloader_size) * 20)
                    print(f"{iteration - st_iter}/{dataloader_size} {'=' * percent}{'-' * (20 - percent)}")
                if phase == 'train':
                    exp_lr_scheduler.step()

                epoch_loss = running_loss / datasets_size[phase]
                epoch_acc = running_corrects.double() / datasets_size[phase]

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 모델을 깊은 복사(deep copy)함
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(net.state_dict())
                    best_epoch = epoch
                    best_iteration = iteration

            print()
    except KeyboardInterrupt:
        if args.interrupt:
            print('Stopping early. Saving network...')

            # Delete previous copy of the interrupted network so we don't spam the weights folder
            SavePath.remove_interrupt(args.save_directory)

            net.save_weights(save_path(epoch, repr(iteration) + '_interrupt'))
        exit()

    # Save default finished model
    net.save_weights(save_path(epoch, iteration))

    # Save Best Accuracy Model
    print(f"Best epoch and iteration is ({best_epoch}, {best_iteration})")
    net.load_state_dict(best_model_wts)
    net.save_weights(save_path(best_epoch, best_iteration))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def data_show():
    data_transform = transforms.Compose([
        transforms.Resize(380),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    total_dataset = torchvision.datasets.ImageFolder(root=args.dataset, transform=data_transform)

    alpha = 0.8
    total_size = len(total_dataset)
    train_size = int(len(total_dataset) * alpha)
    train_dataset, val_dataset = random_split(total_dataset, (train_size, total_size - train_size))
    datasets = {'train': train_dataset,
                'val': val_dataset}
    datasets_size = {'train': train_size,
                     'val': total_size - train_size}
    dataloaders = {'train': DataLoader(train_dataset, batch_size=args.batch_size,
                                       shuffle=True, num_workers=4),
                   'val': DataLoader(val_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4)}
    class_names = total_dataset.classes

    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # 갱신이 될 때까지 잠시 기다립니다.

    # 학습 데이터의 배치를 얻습니다.
    inputs, classes = next(iter(dataloaders['train']))

    # 배치로부터 격자 형태의 이미지를 만듭니다.
    out = torchvision.utils.make_grid(inputs)

    imshow(out, title=[class_names[x] for x in classes])


if __name__ == '__main__':
    main()



