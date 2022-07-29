import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as o
from torch.utils.data import DataLoader, random_split
# import torch.distributed as dist

import mpi4py
import numpy as np
from mpi4py import MPI

import pandas as pd

import cords
from cords.utils.data.datasets.SL import gen_dataset
from cords.utils.data.dataloader.SL.adaptive import GradMatchDataLoader, RandomDataLoader
# from cords.utils.models import ResNet18
from dotmap import DotMap

import deephyper as dh
from deephyper.problem import HpProblem
from deephyper.evaluator import Evaluator, profile
from deephyper.search.hps import CBO
from deephyper.evaluator.callback import TqdmCallback

import pathlib
import os
import os.path as osp
import logging
import sys
import time
import csv

mpi4py.rc.initialize = False
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"

if not MPI.Is_initialized():
    MPI.Init_thread()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

search_log_dir = "search_log/"
pathlib.Path(search_log_dir).mkdir(parents=False, exist_ok=True)

def load_data():
    train_ds, valid_ds, test_ds, num_cls = gen_dataset('/lus/grand/projects/datascience/ianwixom/expcifar/', 'cifar10', None, isnumpy=False)
    
    return train_ds, valid_ds

def importresults():
    data_full = pd.read_csv("/lus/grand/projects/datascience/ianwixom/expcifar/r_full/results.csv") # noncords full length
    data_short = pd.read_csv("/lus/grand/projects/datascience/ianwixom/expcifar/r_short/results.csv") # noncords short length
    data_random = pd.read_csv("/lus/grand/projects/datascience/ianwixom/expcifar/r_random/results.csv") # random full length
    data_grad = pd.read_csv("/lus/grand/projects/datascience/ianwixom/expcifar/r_gradmatch/results.csv") # gradmatch full length

    ### finding top performers in each run
    result_full = rankresults(data_full)
    result_short = rankresults(data_short)
    result_random = rankresults(data_random)
    result_grad = rankresults(data_grad)

    return result_full, result_short, result_random, result_grad

def rankresults(data):
    i_max = data.objective.argmax()
    result = pd.DataFrame(data.iloc[i_max])

    for i in range(8):
        data = data.drop(i_max, axis = 0)
        i_max = data.objective.argmax()
        result = pd.concat([result, pd.DataFrame(data.iloc[i_max])], axis = 1)
        data.index = pd.RangeIndex(len(data.index))

    result = result.T
    result.index = pd.RangeIndex(len(result.index))

    return result

def train(model, criterion, optimizer, scheduler, epochs, dl, valid_dl, device):
    acc_max = 0

    acc = eval(model, criterion, valid_dl, device)
    print(f"The accuracy of the model of worker {rank} on epoch 0 is {round(acc*100, 2)}%")

    for i in range(epochs):
        model.train()

        for _, (features, labels) in enumerate(dl):
            features, labels = features.to(device), labels.to(device, non_blocking = True)
            
            optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            # id = predictions

        if (i % 10 == 0):
            acc = eval(model, criterion, valid_dl, device)
            print(f"The accuracy of the model of worker {rank} on epoch {i+1} is {round(acc*100, 2)}%")
            
            if acc_max < acc:
                acc_max = acc
        scheduler.step()
    
    acc = eval(model, criterion, valid_dl, device)
    print(f"The accuracy of the model of worker {rank} on epoch {epochs} is {round(acc*100, 2)}%")

    if acc_max < acc:
        acc_max = acc
    return round(acc_max, 4)


def eval(model, criterion, dl, device):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for _, (features, labels) in enumerate(dl):
            features, labels = features.to(device), labels.to(device, non_blocking = True)
            predictions = model(features)
            loss = criterion(predictions, labels)
            correct += float((predictions.argmax(1) == labels).type(torch.float).sum().item())
    return correct / len(dl.dataset)


def run(config: dict):
    acc = 0
    device = torch.device("cuda" if is_gpu_available else "cpu")
    if is_gpu_available and n_gpus > 1:
        device = torch.device("cuda", rank)
        print("Running on GPU", rank)
    else:
        device = torch.device("cuda")
        print("Running on the GPU")
    train_ds, valid_ds = load_data()
    train_dl = DataLoader(train_ds, batch_size = 64, shuffle = True, num_workers = 0, pin_memory = True)
    valid_dl = DataLoader(valid_ds, batch_size = 64, shuffle = True, num_workers = 0, pin_memory = True)

    block_struct = [2, 2, 2, 2]

    model = ResNet(BasicBlock, block_struct, 10)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = o.RMSprop(model.parameters(), lr=config["lr"],
                                                momentum=config['momentum'],
                                                weight_decay=config['weightdecay'])
    
    scheduler = o.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    acc = train(model, criterion, optimizer, scheduler, epochs, train_dl, valid_dl, device)

    #Free GPU memory
    del model
    # torch.cuda.empty_cache()

    return acc



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = f.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )


    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = f.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = f.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 8 * self.in_planes * block.expansion
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = f.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = f.avg_pool2d(out, 4)
                e = out.view(out.size(0), -1)
        else:
            out = f.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = f.avg_pool2d(out, 4)
            e = out.view(out.size(0), -1)
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim
def ResNet34(block_struct, num_classes = 10):
    return ResNet(BasicBlock, block_struct, num_classes)

def topaverage(result, portion):
    i_max = 0
    avg_obj = 0

    if (portion % 2 == 0):
        for i in range(4):
            obj_tempmax = 0
            
            obj = run(result.iloc[i])
            obj_tempmax += obj
            if avg_obj < obj_tempmax:
                avg_obj = obj_tempmax
                i_max = i
    else:
        for i in range(4):
            obj_tempmax = 0
            obj = run(result.iloc[i+4])
            obj_tempmax += obj
            if avg_obj < obj_tempmax:
                avg_obj = obj_tempmax
                i_max = i+4

    topavg = np.array([i_max, avg_obj])
    comm.Send([topavg, MPI.INT], dest=0, tag=10)
    file = open('/home/iwixom/ianwixom/expcifar/compareresults.csv', 'a')
    writer = csv.writer(file)
    writer.writerow([i_max, avg_obj, rank])
    file.close()
    # close the file




if __name__ == "__main__":
    is_gpu_available = torch.cuda.is_available()
    n_gpus = torch.cuda.device_count() if is_gpu_available else 0
    epochs = 75

    if rank == 0:
        file = open('/home/iwixom/ianwixom/expcifar/compareresults.csv', 'w')
        writer = csv.writer(file)
        print("Running comparison test across 8 GPUs, organized by dataset structure within code.\n")
        writer.writerow(['location', 'avg_obj', 'rank'])
        file.close()
        
    result_full, result_short, result_random, result_grad = importresults()

    if rank == 0 or rank == 1:
        print("Running full dataset, long epochs on rank {}".format(rank))
        topaverage(result_full, rank)
    
    if rank == 2 or rank == 3:
        print("Running full dataset, short epochs on rank {}".format(rank))
        topaverage(result_short, rank)

    if rank == 4 or rank == 5:
        print("Running random dataset, long epochs on rank {}".format(rank))
        topaverage(result_random, rank)

    if rank == 6 or rank == 7:
        print("Running GradMatch dataset, long epochs on rank {}".format(rank))
        topaverage(result_grad, rank)
    
    print("Finished. Ending GPU run")

    # if rank == 0:
    #     file.close()
    #     time.sleep(120)
    #     results = np.zeros((4,2), dtype=np.float64)
    #     for i in range(4):
    #         result1 = np.empty(2, dtype=np.float64)
    #         result2 = np.empty(2, dtype=np.float64)
    #         comm.Recv(result1, source=i*2, tag=10)
    #         comm.Recv(result2, source=i*2+1, tag=10)
    #         print(result1, result2)
    #         if result1[1] > result2[1]:
    #             results[i, 0] = result1[0]
    #             results[i, 1] = result1[1]
    #         else:
    #             results[i, 0] = result2[0]
    #             results[i, 1] = result2[1]
    #     print(results)
    #     print(f"Results of fully training the top ten from each group and producing its average from 5 runs: \n" \
    #         f"Full Dataset, long epochs: Job ID {results[0,0]} had an average accuracy of {round(results[0,1]*100, 2)}% \n" \
    #         f"Full Dataset, short epochs: Job ID {results[1,0]} had an average accuracy of {round(results[1,1]*100, 2)}% \n" \
    #         f"Random: Job ID {results[2,0]} had an average accuracy of {round(results[2,1]*100, 2)}% \n" \
    #         f"GradMatch: Job ID {results[3,0]} had an average accuracy of {round(results[3,1]*100, 2)}% \n")




    