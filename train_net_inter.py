# -*- coding: utf-8 -*-
# @Author: Chen Renjie
# @Date:   2021-02-28 15:31:30
# @Last Modified by:   Chen Renjie
# @Last Modified time: 2021-02-28 20:45:03
import sys

import os
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.data import DataLoader

from interaction import inter_m_order
from dataset import CIFAR10mini
from utils import Logger, madrys



def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Logistic_trainer:
    def __init__(self, args):
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)
        self.path = {
            "root": args.root,
            "result_path": args.result_path
        }
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = args
        self.loss_type = args.loss_type
        self.lr = args.lr
        self.gamma = args.gamma
        self.bs = args.batchsize
        self.start_epoch = args.start_epoch
        self.epoch_num = args.epoch_num
        self.num_class = args.num_class
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.fine_tune_path = args.fine_tune_path
        self.lam = args.lam
        self.order = args.order
        self.seed_num = args.seed
        self.list = [[], [], [], [], [], []]
        filename = os.path.join(self.path["result_path"], "log.txt")
        if self.start_epoch!=0 and os.path.exists(filename):
            os.remove(filename) 
        self.logger = Logger(filename, level=args.log).logger
        self.save_hparam(args)

    def save_hparam(self, args):
        savepath = os.path.join(self.path["result_path"], "hparam.txt")
        with open(savepath, "w") as f:
            args_dict = args.__dict__
            for key in args_dict:
                f.write(f"{key} : {args_dict[key]}\n")
        self.logger.debug("Save the hyper-parameter!")

    def load_latest_epoch(self):
        model_path = os.path.join(self.path["result_path"], "model.pkl")
        self.model.load_state_dict(torch.load(model_path))
        list_path = os.path.join(self.path["result_path"], "list.pkl")
        self.list = torch.load(list_path)
        self.logger.debug(f"Load lastest epoch({self.start_epoch-1})")

    def save_latest_epoch(self, epoch):
        model_path = os.path.join(self.path["result_path"], "model.pkl")
        torch.save(self.model.state_dict(), model_path)
        list_path = os.path.join(self.path["result_path"], "list.pkl")
        torch.save(self.list, list_path)
        self.logger.debug(f"Save lastest epoch({epoch})")

    def prepare_model(self):
        if self.args.model == "resnet18":
            self.model = torchvision.models.resnet18(pretrained=False, num_classes=self.num_class)
        elif self.args.model == "resnet34":
            self.model = torchvision.models.resnet34(pretrained=False, num_classes=self.num_class)
        elif self.args.model == "resnet50":
            self.model = torchvision.models.resnet50(pretrained=False, num_classes=self.num_class)
        else:
            self.logger.error(f"Unknown model name: {self.args.model}")
            raise ValueError(f"Unknown model name: {self.args.model}")

        if self.start_epoch != 0:
            self.load_latest_epoch()
            self.start_epoch = len(self.list[0])
        if self.fine_tune_path:
            self.model.load_state_dict(torch.load(self.fine_tune_path))

        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        if self.args.opt.upper() == "SGD":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.logger.debug(f"Model {self.args.model}, Optimizer {self.args.opt} used")

    def prepare_dataset(self, **kwargs):
        self.path["data_path"] = kwargs["data_path"]
        train_transform = kwargs["train_transform"]
        train_shuffle = kwargs["train_shuffle"]
        test_transform = kwargs["test_transform"]
        test_shuffle = kwargs["test_shuffle"]
        if self.args.dataset == "CIFAR10":
            train_set = datasets.CIFAR10(root=self.path["data_path"], train=True, download=True, transform=train_transform)
            test_set = datasets.CIFAR10(root=self.path["data_path"], train=False, download=True, transform=test_transform)
        elif self.args.dataset == "cifar-10-mini":
            train_set = CIFAR10mini(root='./data', train=True, transform=train_transform, num_per_class=self.args.train_per_class)
            test_set = CIFAR10mini(root='./data', train=False, transform=test_transform, num_per_class=self.args.test_per_class)
        else:
            train_set = datasets.ImageFolder(root=self.path["data_path"] + "/train", transform=self.train_transform)
            test_set = datasets.ImageFolder(root=self.path["data_path"] + "/test", transform=self.test_transform)
        self.train_loader = DataLoader(train_set, batch_size=self.bs, shuffle=train_shuffle, num_workers=4)
        self.test_loader = DataLoader(test_set, batch_size=self.bs, shuffle=test_shuffle, num_workers=4)

        self.logger.debug(f"Dataset {self.args.dataset} loaded.")

    def get_learning_rate(self):
        lr = []
        for param_group in self.optimizer.param_groups:
            lr += [param_group['lr']]
        return lr

    def get_error(self, pre, lbl):
        acc = torch.sum(pre == lbl).item() / len(pre)
        return 1 - acc

    def train_DNN_raw(self):
        self.model.train()
        Loss, Error = 0, 0
        for i, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            output = self.model(images)
            loss = self.criterion(output, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = output.detach().argmax(dim=1)
            error = self.get_error(preds, labels)

            Error += error
            Loss += loss.cpu().item()
            if (i%10==0):
                self.logger.info(f"[Train raw] batch {i+1:3d}: loss: {Loss / (i + 1):0.3f}, Error: {Error / (i + 1):0.3f}")

    def test_DNN_raw(self):
        self.model.eval()
        Loss, Error = 0, 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(self.device), labels.to(self.device)

                output = self.model(images)
                loss = self.criterion(output, labels)

                preds = output.detach().argmax(dim=1)
                error = self.get_error(preds, labels)

                Error += error
                Loss += loss.cpu().item()

                if (i%10==0):
                    self.logger.info(f"[Test raw] batch {i+1:3d}: loss: {Loss / (i + 1):0.3f}, Error: {Error / (i + 1):0.3f}")
        self.list[4].append(Loss / (i + 1))
        self.list[5].append(Error / (i + 1))



    def train_DNN(self):
        self.model.train()
        # torch.autograd.set_detect_anomaly(True)
        Loss, Loss_ce, Loss_inter, Error = 0, 0, 0, 0
        Loss_inter_ori, Loss_inter_adv = 0, 0
        for i, (img, lbl) in enumerate(self.train_loader):
            img = img.to(self.device)          # shape: torch.Size([64, 3, 32, 32])
            lbl = lbl.to(self.device)
            img_adv = madrys(self.model, img, lbl, self.device, step_size = 2/255, epsilon= 8/255, perturb_steps=5, isnormalize=False)
            self.model.train()
            output = self.model(img_adv)
            loss_ce = self.criterion(output, lbl)
            if self.loss_type==0:
                loss = loss_ce
            else:
                loss_inter_adv = inter_m_order(self.args, self.model, img_adv, lbl, self.logger)
                Loss_inter_adv += loss_inter_adv.mean().cpu().item()
                if self.loss_type==1:
                    loss = loss_ce + self.lam * loss_inter_adv.mean()
                elif self.loss_type==2:
                    loss_inter_img = inter_m_order(self.args, self.model, img, lbl, self.logger)
                    loss_inter = (torch.sqrt((loss_inter_adv - loss_inter_img) ** 2)).mean()
                    loss = loss_ce + self.lam * loss_inter    # loss 2
                    Loss_inter_ori += loss_inter_img.mean().cpu().item()
                    Loss_inter += loss_inter.cpu().item()
                else:
                    raise ValueError("Unknown Loss type.")
                self.model.train()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            pre = output.detach().argmax(dim=1)
            error = self.get_error(pre, lbl)

            Error += error
            Loss += loss.cpu().item()
            Loss_ce += loss_ce.cpu().item()

            if (i%5==0):
                self.logger.info(f"[Train] batch {i+1:3d}: inter_ori: {Loss_inter_ori / (i + 1):0.3f}, inter_adv: {Loss_inter_adv / (i + 1):0.3f}, loss_inter: {Loss_inter / (i + 1):0.3f}, loss_ce: {Loss_ce / (i + 1):0.3f}, Loss: {Loss / (i + 1):0.3f} Error: {Error / (i + 1):0.3f}")

        self.list[0].append(Loss / (i + 1))
        self.list[1].append(Error / (i + 1))


    def test_DNN(self):
        self.model.eval()

        Loss, Loss_ce, Loss_inter, Error = 0, 0, 0, 0
        Loss_inter_ori, Loss_inter_adv = 0, 0
        with torch.no_grad():
            for i, (img, lbl) in enumerate(self.test_loader):
                img = img.to(self.device)
                lbl = lbl.to(self.device)
                img_adv = madrys(self.model, img, lbl, self.device, step_size = 2/255, epsilon= 8/255, perturb_steps=5, isnormalize=False)

                output = self.model(img_adv)
                loss_ce = self.criterion(output, lbl)
                if self.loss_type==0:
                    loss = loss_ce
                else:
                    loss_inter_adv = inter_m_order(self.args, self.model, img_adv, lbl, self.logger)
                    Loss_inter_adv += loss_inter_adv.mean().cpu().item()
                    if self.loss_type==1:
                        loss = loss_ce + self.lam * loss_inter_adv.mean()
                    elif self.loss_type==2:
                        loss_inter_img = inter_m_order(self.args, self.model, img, lbl, self.logger)
                        loss_inter = (torch.sqrt((loss_inter_adv - loss_inter_img) ** 2)).mean()
                        loss = loss_ce + self.lam * loss_inter    # loss 2
                        Loss_inter_ori += loss_inter_img.mean().cpu().item()
                        Loss_inter += loss_inter.cpu().item()
                    else:
                        raise ValueError("Unknown Loss type.")
                    self.model.train()

                pre = output.detach().argmax(dim=1)
                error = self.get_error(pre, lbl)

                Error += error
                Loss += loss.cpu().item()
                Loss_ce += loss_ce.cpu().item()

                if (i%5==0):
                    self.logger.info(f"[test] batch {i+1:3d}: inter_ori: {Loss_inter_ori / (i + 1):0.3f}, inter_adv: {Loss_inter_adv / (i + 1):0.3f}, loss_inter: {Loss_inter / (i + 1):0.3f}, loss_ce: {Loss_ce / (i + 1):0.3f}, Loss: {Loss / (i + 1):0.3f} Error: {Error / (i + 1):0.3f}")

        self.list[2].append(Loss / (i + 1))
        self.list[3].append(Error / (i + 1))


    def draw_figure(self):
        x = np.arange(0, len(self.list[0]), 1)
        train_l, train_e = np.array(self.list[0]), np.array(self.list[1])
        test_l, test_e = np.array(self.list[2]), np.array(self.list[3])
        test_lr, test_er = np.array(self.list[4]), np.array(self.list[5])
        plt.figure()
        plt.subplot(211)
        plt.plot(x, train_l, color='C0', label="Train")
        plt.plot(x, test_l, color='C1', label="Test")
        plt.plot(x, test_lr, color='C2', label="Test raw")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.subplot(212)
        plt.plot(x, train_e, color='C0', label="Train")
        plt.plot(x, test_e, color='C1', label="Test")
        plt.plot(x, test_er, color='C2', label="Test raw")
        plt.xlabel("epoch")
        plt.ylabel("error")
        plt.legend()

        plt.savefig(os.path.join(self.path["result_path"], "curve.png"))
        plt.close()

    def print_and_save_list(self):
        save_path = os.path.join(self.path["result_path"], "list.txt")
        with open(save_path, "w") as f:
            train_loss = self.list[0]
            train_error = self.list[1]
            test_loss = self.list[2]
            test_error = self.list[3]
            test_loss_raw = self.list[4]
            test_error_raw = self.list[5]

            for i in range(len(train_loss)):
                f.write(f"Epoch{i+1:3d} train_loss:{train_loss[i]:0.3f}, train_error:{train_error[i]:0.3f}, test_loss:{test_loss[i]:0.3f}, test_error:{test_error[i]:0.3f}, test_loss_raw:{test_loss_raw[i]:0.3f}, test_error_raw:{test_error_raw[i]:0.3f}\n")

    def draw_parameters(self):
        self.path['distribution_path'] = os.path.join(self.path['result_path'], "distribution")
        if not os.path.exists(self.path['distribution_path']):
            os.makedirs(self.path['distribution_path'])
        for p in self.model.named_parameters():
            name = p[0].split(" Parameter containing:")[0].replace(".", "_") + ".jpg"
            fig_path = os.path.join(self.path['distribution_path'], name)
            plt.hist(torch.flatten(p[1]).detach().cpu().numpy())
            plt.savefig(fig_path)
            plt.close()

    def work(self):
        if self.loss_type!=0 and self.start_epoch==0:
            for _ in range(5):
                self.train_DNN_raw()
                # self.test_DNN_raw()
        for epoch in range(self.start_epoch, self.epoch_num):
            self.logger.debug(f"Epoch {epoch} start...")
            seed_torch(epoch + args.batchsize)
            self.lr = self.get_learning_rate()
            self.train_DNN()
            self.test_DNN()
            self.test_DNN_raw()
            if (epoch%self.args.save_epoch == 0) or (epoch == self.epoch_num-1):
                self.save_latest_epoch(epoch)
            self.draw_figure()
            self.print_and_save_list()
        self.draw_figure()
        self.print_and_save_list()
        self.draw_parameters()


if __name__ == "__main__":
    """
    [usage]:
    $ cd defense
    $ pwd
    $ ./defense
    $ python train_net_inter.py    # train with default paramters
    $ python train_net_inter.py --start_epoch 0 --epoch_num 20
    [File directory]:
    exp
    |-- data    # 数据集目录
    | |-- CIFAR10
    | |-- ImageNet
    | |-- ImageFolder    # 自定义数据集
    |   |-- train    # 数据训练集目录
    |   |-- test    # 数据测试集目录
    |-- result    # 实验结果
    | |-- CIFAR10_resnet18    # 数据集+模型结果
    |   |-- distribution    # 参数可视化
    |     |-- bn1_bias.jpg
    |     |-- bn1_weight.jpg
    |     |-- ......
    |     |-- layer4_1_conv2_weight.jpg
    |   |-- curve.png    # 损失和错误率曲线
    |   |-- hparam.txt    # 超参数存储
    |   |-- list.txt    # 损失和错误率记录
    |   |-- log.txt    # 日志文件
    |   |-- list_xx.pkl    # list_epoch
    |   |-- model_xx.pkl    # epoch模型参数
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="resnet18", choices=["resnet18", "resnet34", "resnet50"], type=str, help="Backbone Network {Res-18, Res-34, Res-50}.")
    parser.add_argument("-d", "--dataset", default="cifar-10-mini", type=str, help="Dataset.")   # "CIFAR10"
    parser.add_argument("--train_per_class", default=500, type=int, help="train num per class")
    parser.add_argument("--test_per_class", default=100, type=int, help="test num per class")
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument("--log", choices=["debug", "info"], default="debug", help='Log print level.')
    parser.add_argument("--root", default='./', type=str)
    parser.add_argument("--fine_tune_path", default=None, type=str)
    parser.add_argument("--loss_type", choices=[0, 1, 2], default=0, type=int)
    parser.add_argument("--num_class", default=10, type=int, help="Number of image classes")
    parser.add_argument("--epoch_num", default=50, type=int, help="Number of Epochs")        # 50
    parser.add_argument("--start_epoch", default=0, type=int, help="Train start from # epochs")
    parser.add_argument("--save_epoch", default=1, type=int, help="Save model every # epochs")
    parser.add_argument("--batchsize", default=2, type=int, help="Batch size")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--opt", default="SGD", type=str)
    parser.add_argument("--seed", default=0, type=int, help="random seed")
    parser.add_argument("--lam", type=float, default=-0.1)
    parser.add_argument("--order", type=float, default=0.95)

    parser.add_argument("--grid-size", default=8, type=int)
    parser.add_argument("--img-size", default=32, type=int)
    parser.add_argument("--pair_num", default=50, type=int, help='number of point pair of each test img')    # 50
    parser.add_argument("--sample_num", default=32, type=int, help='sample num of S')
    parser.add_argument("--ratios", default=[0.95], type=list, help='ratios of context') #[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1]
    parser.add_argument("--cal_batch", default=80, type=int, help='calculate # of images per forward')
    parser.add_argument('--softmax_type', default='normal', type=str)

    args = parser.parse_args()
    seed_torch(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

    args.result_path = os.path.join(args.root, "result", f"{args.dataset}_{args.model}_{args.loss_type}_{args.lam}")
    if args.fine_tune_path:
        args.result_path = args.result_path + "_finetune"

    if args.dataset == "imagenet" or args.dataset == "tiny-imagenet-200":
        img_size = 224
    elif args.dataset == "CIFAR10" or args.dataset == "cifar-10-mini":
        img_size = 32
    args.data_path = os.path.join(args.root, "data", args.dataset)

    train_transform = transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainer = Logistic_trainer(args)
    trainer.prepare_model()
    trainer.prepare_dataset(data_path=args.data_path, train_transform=train_transform, train_shuffle=True, test_transform=test_transform, test_shuffle=True)

    trainer.work()
