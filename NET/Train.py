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


def train(model, loader, criterion, optimizer, device, shed):
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

        output = model(data)

        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Training finished, total time = {}".format(finish - start))


def test(model1, loader1, model2, loader2, device):
    model1.eval()
    model2.eval()

    with torch.no_grad():
        correct_p = 0
        correct_v = 0

        for data, target in loader1:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            policy = model1(data)

            policy = F.softmax(policy, dim=1)

            target = target.to(device)

            pred = policy.data.max(1, keepdim=True)[1]
            correct_p += int(pred.eq(target.data.view_as(pred)).sum())

        for data, target in loader2:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            v = model2(data)

            v = F.softmax(v, dim=1)

            target = target.to(device)

            pred = v.data.max(1, keepdim=True)[1]
            correct_v += int(pred.eq(target.data.view_as(pred)).sum())

        print("Policy {:.3f} ".format(correct_p / len(loader1.dataset)) + "V {:.3f}".format(
            correct_v / len(loader2.dataset)))


def main(epoches_num):
    gc.enable()
    print("Available cudas {}\n".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device, '\n')

    model1 = Net()
    model2 = VNet()

    if torch.cuda.device_count() > 1:
        model1 = torch.nn.DataParallel(model1)
        print("Using {} cudas for our model".format(torch.cuda.device_count()))
        model2 = torch.nn.DataParallel(model2)
        print("Using {} cudas for our model\n".format(torch.cuda.device_count()))

    path = "model_{}{}.pth"

    loss1 = torch.nn.CrossEntropyLoss()
    loss2 = torch.nn.CrossEntropyLoss()

    loss1 = loss1.to(device)
    loss2 = loss2.to(device)

    lr = 0.01

    optimizer1 = torch.optim.Adam(model1.parameters(), lr=lr)
    shed1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=10, gamma=0.5)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=lr)
    shed2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=10, gamma=0.5)

    for number in range(epoches_num):
        print("Start epoch {}".format(number))

        if os.path.isfile(path.format('p', number)):
            model1.load_state_dict(torch.load(path.format('p', number), map_location=lambda storage, loc: storage))
            print("Model p loaded")
        if os.path.isfile(path.format('v', number)):
            model2.load_state_dict(torch.load(path.format('v', number), map_location=lambda storage, loc: storage))
            print("Model v loaded\n")

        model1 = model1.to(device)
        model2 = model2.to(device)

        start = time.clock()

        count = 10

        torch.cuda.empty_cache()

        dataset1, dataset2 = make_dataset(count * number, count * (number + 1))

        print('Datasets ready, size {}, {}, time {}'.format(len(dataset1), len(dataset2), time.clock() - start))

        loader1 = DataLoader(dataset1, batch_size=4096, shuffle=True, drop_last=False, pin_memory=True, num_workers=16)
        loader2 = DataLoader(dataset2, batch_size=4096, shuffle=True, drop_last=False, pin_memory=True, num_workers=16)

        del dataset1
        del dataset2

        torch.cuda.empty_cache()

        train(model1, loader1, loss1, optimizer1, device, shed1)
        train(model2, loader2, loss2, optimizer2, device, shed2)

        if number % 10 == 0:
            test(model1, loader1, model2, loader2, device)

        del loader1
        del loader2

        torch.cuda.empty_cache()

        torch.save(model1.state_dict(), path.format('p', number + 1))
        torch.save(model2.state_dict(), path.format('v', number + 1))

        print("Finish epoch {}\n".format(number))


main(300)
