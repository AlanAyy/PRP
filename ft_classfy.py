import os
import time
import random

import numpy as np
import torch
from torch import nn, optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, random_split

import argparse
from pprint import pprint
from tqdm import tqdm

from models import c3d, r3d, r21d
from datasets.predict_dataset import ClassifyDataSet
from config import params

TRAIN_MODE = "train"
EVAL_MODE = "eval"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Training:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # Initialize device
        if self.args.device == "best":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = self.args.device

        # Initialize data
        self.train_loader, self.val_loader = self._init_dataloaders()

        # Initialize model
        self.model = self._init_model()
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Initialize save data
        save_path = f"{self.args.save_path}ft_classify_{self.args.exp_name}_{self.args.dataset_type}"
        self.model_save_dir = os.path.join(save_path, time.strftime("%m-%d-%H-%M"))
        self.writer = SummaryWriter(self.model_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        # Initialize optimizer and scheduler
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        if self.args.num_classes == 101:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=100, gamma=0.1
            )
        elif self.args.num_classes == 3:
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=6, gamma=0.1
            )
        else:
            raise ValueError("num_classes must be 101 or 3")

    def _init_model(self) -> nn.Module:
        if self.args.model_type == "c3d":
            model = c3d.C3D(with_classifier=True, num_classes=self.args.num_classes)
        elif self.args.model_type == "r3d":
            model = r3d.R3DNet((1, 1, 1, 1), with_classifier=True, num_classes=self.args.num_classes)
        elif self.args.model_type == "r21d":
            model = r21d.R2Plus1DNet((1, 1, 1, 1), with_classifier=True, num_classes=self.args.num_classes)

        model.load_state_dict(self._load_weights(self.args.pretrain_path), strict=False)

        # Uncomment below to train only the last layer
        # model.linear = nn.Linear(512, 101)
        # for name, param in model.named_parameters():
        #     param.requires_grad = True if 'linear' in name else False

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)

        model.to(self.device)
        return model

    def _load_weights(self, pretrained_weights_path: str) -> dict:
        if pretrained_weights_path == "" or not os.path.exists(pretrained_weights_path):
            raise ValueError("pretrained_weights_path cannot be empty and must exist")

        adjusted_weights = {}
        pretrained_weights = torch.load(pretrained_weights_path, map_location="cpu")
        try:
            items = pretrained_weights["model_state_dict.items()"]
        except KeyError:
            items = pretrained_weights.items()

        # Get rid of "module.base_network." in the name
        for name, params in items:
            if "module.base_network." in name:
                name = name[name.find(".") + 14 :]
                adjusted_weights[name] = params
        return adjusted_weights

    def _init_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        train_dataset = ClassifyDataSet(
            self.args.dataset_path,
            mode="train",
            split=self.args.dataset_split,
            data_name=self.args.dataset_type,
        )

        if "ucf" in self.args.dataset_type.lower():
            val_size = 800
        elif "hmdb" in self.args.dataset_type.lower():
            val_size = 400
        else:
            raise ValueError("data parameter must be 'ucf' or 'hmdb'")

        train_dataset, val_dataset = random_split(
            train_dataset, (len(train_dataset) - val_size, val_size)
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
        )
        return train_loader, val_loader

    def run_epoch(self, epoch: int, mode: str) -> tuple[float, float]:
        if mode == TRAIN_MODE:
            loader = self.train_loader
        elif mode == EVAL_MODE:
            loader = self.val_loader
            total_loss = 0.0
        else:
            raise ValueError("mode must be 'train' or 'eval'")

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        for step, (inputs, labels) in enumerate(tqdm(loader, leave=False)):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Compute output
            output = self.model(inputs)
            loss = self.criterion(output, labels)
            losses.update(loss.item(), inputs.size(0))

            # Measure accuracy and record loss
            if self.args.num_classes > 5:
                prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))
            else:
                prec1 = accuracy(output.data, labels, topk=(1,))[0]
                top1.update(prec1.item(), inputs.size(0))

            if mode == TRAIN_MODE:
                # Compute gradient and step optimizer
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            elif mode == EVAL_MODE:
                total_loss += loss.item()

            if (step + 1) % self.args.print_freq == 0:
                self.pretty_print(self.train_loader, epoch, step, losses, top1, top5)

        if mode == EVAL_MODE:
            avg_loss = total_loss / len(loader)
            return avg_loss, top1.avg

    def train(self) -> None:
        # Initialize training variables
        start_epoch = 1
        best_epoch = 0
        self.best_acc = 0
        self.best_val_loss = float("inf")

        for epoch in tqdm(range(start_epoch, self.args.epoch_num + 1)):
            # Train
            self.model.train()
            self.run_epoch(epoch, mode=TRAIN_MODE)

            # Validate
            self.model.eval()
            with torch.no_grad():
                val_loss, top1_avg = self.run_epoch(epoch, mode=EVAL_MODE)

            # Update scheduler
            self.scheduler.step()

            # Save the model
            self.model_saver(epoch, val_loss, top1_avg)

        print(f"Training complete! Best accuracy: {self.best_acc:.3f} at epoch {best_epoch}")

    def pretty_print(self, loader: DataLoader, epoch: int, step: int, losses: AverageMeter, top1: AverageMeter, top5: AverageMeter) -> None:
        print("\n-----------------------------------------------")
        for param in self.optimizer.param_groups:
            print(f"lr: {param['lr']}")

        p_str = f"Epoch: [{epoch}][{step + 1}/{len(loader)}]"
        print(p_str)

        p_str = f"Loss: {losses.avg:.5f}"
        print(p_str)

        total_step = (epoch - 1) * len(loader) + step + 1
        self.writer.add_scalar("train/loss", losses.avg, total_step)
        self.writer.add_scalar("train/acc", top1.avg, total_step)

        if self.args.num_classes > 5:
            p_str = f"Top-1 accuracy: {top1.avg:.2f}%, Top-5 accuracy: {top5.avg:.2f}%"
        else:
            p_str = f"Top-1 accuracy: {top1.avg:.2f}%"
        print(p_str)

    def model_saver(self, epoch: int, val_loss: int, top1_avg: int) -> None:
        model_path = None

        if epoch % self.args.save_freq == 0:
            model_path = os.path.join(
                self.model_save_dir, f"ckpt_model_{epoch}.pth.tar"
            )

        if top1_avg > self.best_acc and val_loss < self.best_val_loss:
            print(f"\nNew best model! Accuracy: {top1_avg:.3f}, Loss: {val_loss:.3f} ")
            self.best_acc = top1_avg
            self.best_val_loss = val_loss
            model_path = os.path.join(
                self.model_save_dir, f"best_both_model_{epoch}.pth.tar"
            )
        elif top1_avg > self.best_acc:
            print(f"\nNew best accuracy! Accuracy: {top1_avg:.3f}, Loss: {val_loss:.3f}")
            self.best_acc = top1_avg
            model_path = os.path.join(
                self.model_save_dir, f"best_acc_model_{epoch}.pth.tar"
            )
        elif val_loss < self.best_val_loss:
            print(f"\nNew best loss! Accuracy: {top1_avg:.3f}, Loss: {val_loss:.3f}")
            self.best_val_loss = val_loss
            model_path = os.path.join(
                self.model_save_dir, f"best_loss_model_{epoch}.pth.tar"
            )

        if model_path is not None:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "best_acc": self.best_acc,
                    "best_val_loss": self.best_val_loss,
                },
                model_path,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video Clip Restruction and Playback Rate Prediction")
    # Data parameters
    parser.add_argument("--exp_name", type=str, default=params["exp_name"], help="experiment name")
    parser.add_argument("--dataset_type", type=str, default=params["dataset_type"], help="dataset type")
    parser.add_argument("--dataset_path", type=str, default=params["dataset_path"], help="dataset path")
    parser.add_argument("--dataset_split", type=str, default=params["dataset_split"], help="dataset split number")
    parser.add_argument("--num_classes", type=int, default=params["num_classes"], help="number of classes")
    parser.add_argument("--save_path", type=str, default=params["save_path_base"], help="save path base")
    parser.add_argument("--pretrain_path", type=str, default=params["pretrained_weights_path"], help="pretrained model path")
    # Training parameters
    parser.add_argument("--epoch_num", type=int, default=params["epoch_num"], help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=params["batch_size"], help="batch size")
    parser.add_argument("--step", type=int, default=params["step"], help="step size")
    # Model parameters
    parser.add_argument("--model_type", type=str, default=params["model_type"], help="model type")
    parser.add_argument("--device", type=str, default=params["device"], help="device")
    # Hyperparameters
    parser.add_argument("--lr", type=float, default=params["learning_rate"], help="learning rate")
    parser.add_argument("--momentum", type=float, default=params["momentum"], help="momentum")
    parser.add_argument("--weight_decay", type=float, default=params["weight_decay"], help="weight decay")
    # Miscellaneous
    parser.add_argument("--gpu", type=str, default=params["gpu"], help="GPU id")
    parser.add_argument("--num_workers", type=int, default=params["num_workers"], help="number of workers")
    parser.add_argument("--print_freq", type=int, default=params["print_freq"], help="print frequency")
    parser.add_argument("--save_freq", type=int, default=params["save_freq"], help="save frequency")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    pprint(vars(args))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    training = Training(args)
    training.train()


if __name__ == "__main__":
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    main()
