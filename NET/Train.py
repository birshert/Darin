from parser import *
from Net import *
import torch
import torch.nn
import torch.optim
import torch.cuda
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import os
import gc


def train_policy(model, loader, criterion, optimizer, device, shed):
    start = time.clock()

    model.train()
    shed.step()

    for data, target in loader:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.type(torch.cuda.FloatTensor)
        else:
            data = data.type(torch.FloatTensor)

        target = target.to(device)

        policy = model(data)

        loss = criterion(policy, target)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Policy training finished, total time = {}".format(finish - start))


def train_v(model, loader, criterion, optimizer, device, shed):
    start = time.clock()

    model.train()
    shed.step()

    for data, target in loader:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.type(torch.cuda.FloatTensor)
        else:
            data = data.type(torch.FloatTensor)

        target = target.to(device)

        v = model(data)

        loss = criterion(v, target)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("V training finished, total time = {}\n".format(finish - start))


def test(modelp, modelv, p_loader, v_loader, device):
    modelp.eval()
    modelv.eval()

    with torch.no_grad():
        correct_p = 0
        correct_v = 0

        for data, target in p_loader:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            policy = modelp(data)

            policy = F.softmax(policy, dim=1)

            target = target.to(device)

            pred = policy.data.max(1, keepdim=True)[1]
            correct_p += int(pred.eq(target.data.view_as(pred)).sum())

        for data, target in v_loader:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            v = modelv(data)

            v = F.softmax(v, dim=1)

            target = target.to(device)

            pred = v.data.max(1, keepdim=True)[1]
            correct_v += int(pred.eq(target.data.view_as(pred)).sum())

        print("Policy {}/{} ({:.3f}%) V {}/{} ({:.3f}%)".format(correct_p, len(p_loader.dataset),
                                                                100.0 * correct_p / len(p_loader.dataset),
                                                                correct_v, len(v_loader.dataset),
                                                                100.0 * correct_v / len(v_loader.dataset)))


def main(epoches_num, sub_epoches_num):
    gc.enable()
    print("Available cudas {}\n".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device, '\n')

    modelp = PNet()
    modelv = VNet()

    if torch.cuda.device_count() > 1:
        modelp = torch.nn.DataParallel(modelp)
        print("Using {} cudas for our P model\n".format(torch.cuda.device_count()))
        modelv = torch.nn.DataParallel(modelv)
        print("Using {} cudas for our V model\n".format(torch.cuda.device_count()))

    path = "model_{}{}.pth"

    for number in range(epoches_num):
        print("Start epoch {}".format(number))

        if os.path.isfile(path.format('p', number)):
            modelp.load_state_dict(torch.load(path.format('p', number), map_location=lambda storage, loc: storage))
            print("Model p loaded")

        if os.path.isfile(path.format('v', number)):
            modelv.load_state_dict(torch.load(path.format('v', number), map_location=lambda storage, loc: storage))
            print("Model v loaded\n")

        modelp = modelp.to(device)
        modelv = modelv.to(device)

        start = time.clock()

        count = 60000

        dataset_p = make_dataset_p(count * number, count * (number + 1))
        dataset_v = make_dataset_v(count * number, count * (number + 1))

        print('Datasets ready, size {}, {}, time {}'.format(len(dataset_p), len(dataset_v), time.clock() - start))

        loader_p = DataLoader(dataset_p, batch_size=4096, shuffle=True, drop_last=False, pin_memory=True,
                              num_workers=torch.cuda.device_count() * 4)

        loader_v = DataLoader(dataset_v, batch_size=4096, shuffle=True, drop_last=False, pin_memory=True,
                              num_workers=torch.cuda.device_count() * 4)

        del dataset_p
        del dataset_v

        torch.cuda.empty_cache()

        loss = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)

        optimizer1 = torch.optim.Adam(modelp.parameters(), lr=0.001)
        optimizer2 = torch.optim.Adam(modelv.parameters(), lr=0.001)
        shed1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=4, gamma=0.15)
        shed2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=5, gamma=0.1)

        test(modelp, modelv, loader_p, loader_v, device)

        for i in range(sub_epoches_num):
            print("SubEpoch {}".format(i))
            train_policy(modelp, loader_p, loss, optimizer1, device, shed1)
            train_v(modelv, loader_v, loss, optimizer2, device, shed2)

        test(modelp, modelv, loader_p, loader_v, device)

        del loader_p
        del loader_v

        torch.cuda.empty_cache()

        torch.save(modelp.state_dict(), path.format('p', number + 1))
        torch.save(modelv.state_dict(), path.format('v', number + 1))

        print("Finish epoch {}\n".format(number))


main(33, 15)
