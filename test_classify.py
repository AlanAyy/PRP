from config import params
from torch.utils.data import DataLoader
from torch import nn, optim
import os
from models import c3d,r3d,r21d
from datasets.predict_dataset import PredictDataset,ClassifyDataSet
import random
import numpy as np
import torch
from tqdm import tqdm

save_path="train_classify"
gpu=0
device_ids = [1]
# torch.cuda.set_device(gpu)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
# params['batch_size'] = 8
# params['num_workers'] = 4
# params['data']='UCF-101'
# params['dataset'] = '/home/Dataset/UCF-101-origin'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device('cpu')

# PRETRAINED_MODEL_PATH = "D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/small_test/10.pth.tar"
PRETRAINED_MODEL_PATH = "D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/ft_classify_default_UCF-101/05-20-15-00/best_acc_model_150.pth.tar"
# PRETRAINED_MODEL_PATH = "D:/Projects/ai-research-school/PRP-video-pace/PRP/outputs/pretrained/best_model_283.pth.tar"

SPLIT_NUMBER = '1'
PRINT_FREQUENCY = 20

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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
def test(test_loader, model, criterion, pretrain_path):
    torch.set_grad_enabled(False)
    model.eval()

    total_loss = 0.0
    correct = 0
    top1 = AverageMeter()

    for step, (inputs,labels) in enumerate(tqdm(test_loader)):
        labels = labels.to(DEVICE)
        inputs = inputs.to(DEVICE)
        outputs = [];
        for clip in inputs:
            clip = clip.to(DEVICE);
            out = model(clip);
            out = torch.mean(out, dim=0)

            outputs.append(out)
        outputs = torch.stack(outputs)
        # print(outputs)

        loss = criterion(outputs, labels)
        # compute loss and acc
        total_loss += loss.item()

        pts = torch.argmax(outputs, dim=1)
        correct += torch.sum(labels == pts).item()
        # print('correct: {}, {}, {}'.format(correct, targets, pts))
        # print(str(step), len(test_loader))
        # print(correct)

        if (step + 1) % PRINT_FREQUENCY == 0 or (step + 1) == len(test_loader):
            print('\nStep: {} | Loss: {:.4f} | Acc: {:.4f} ({}/{})'.format(step + 1, loss.item(), correct / ((step + 1) * params['batch_size']), correct, (step + 1) * params['batch_size']))   

    avg_loss = total_loss / len(test_loader)
    # avg_loss = total_loss / (len(val_loader)+len(train_loader))
    avg_acc = correct / len(test_loader.dataset)
    # avg_acc = correct / (len(val_loader.dataset)+len(train_loader.dataset))
    print(pretrain_path)
    print('[TEST] loss: {:.3f}, acc: {:.3f}'.format(avg_loss, avg_acc))
    return avg_loss

def load_pretrained_weights(ckpt_path):
    adjusted_weights = {}
    pretrained_weights = torch.load(ckpt_path, map_location='cpu')

    try:
        items = pretrained_weights['model_state_dict'].items()
    except KeyError:
        items = pretrained_weights.items()

    for name, params in items:
        # print(name)
        # if "base_network" in name:
        #     name = name[name.find('.')+1:]
        # if "module" in name:
        #     name = name[name.find('.') + 1:]
        if "module.base_network" in name:
            name = name[name.find('.') + 14:]
            # adjusted_weights[name]=params;
        adjusted_weights[name]=params;
    return adjusted_weights;


def test_model(model, pretrain_path):
    print(pretrain_path)
    pretrain_weight = load_pretrained_weights(pretrain_path)
    model.load_state_dict(pretrain_weight,strict= True)
#     model.load_state_dict(torch.load(pretrain_path, map_location='cpu'), strict=True)
    test_dataset = ClassifyDataSet(params['dataset'], mode="test", split=SPLIT_NUMBER, data_name=params['data'])
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                             num_workers=params['num_workers'])

    if len(device_ids)>1:
        print(torch.cuda.device_count())
        model = nn.DataParallel(model)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    test(test_loader, model, criterion,pretrain_path)


if __name__ == '__main__':
    print(1)
    seed = 632
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model=c3d.C3D(with_classifier=True, num_classes=101);
#     model=r3d.R3DNet((1,1,1,1),with_classifier=True, num_classes=101)
#     model=r21d.R2Plus1DNet((1,1,1,1),with_classifier=True, num_classes=101)

    test_model(model, PRETRAINED_MODEL_PATH)
