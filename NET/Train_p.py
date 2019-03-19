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

        policy = model(data)

        loss = criterion(policy, target)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Policy training finished, total time = {}".format(finish - start))


def test(model, loader, device):
    model.eval()

    with torch.no_grad():
        correct = 0

        for data, target in loader:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            v = model(data)

            v = F.softmax(v, dim=1)

            target = target.to(device)

            pred = v.data.max(1, keepdim=True)[1]
            correct += int(pred.eq(target.data.view_as(pred)).sum())

        print("Policy {}/{} ({:.3f}%)".format(correct, len(loader.dataset), 100.0 * correct / len(loader.dataset)))


def main(epoches_num, sub_epoches_num):
    gc.enable()
    print("Available cudas {}\n".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device, '\n')

    model = PNet()

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print("Using {} cudas for our P model\n".format(torch.cuda.device_count()))

    path = "model_{}{}.pth"

    for number in range(epoches_num):
        print("Start epoch {}".format(number))

        if os.path.isfile(path.format('p', number)):
            model.load_state_dict(torch.load(path.format('p', number), map_location=lambda storage, loc: storage))
            print("Model p loaded\n")

        model = model.to(device)

        start = time.clock()

        count = 60000

        dataset = make_dataset_p(count * number, count * (number + 1))

        print('Dataset ready, size {}, time {}'.format(len(dataset), time.clock() - start))

        loader = DataLoader(dataset, batch_size=2048, shuffle=True, drop_last=False, pin_memory=True,
                            num_workers=torch.cuda.device_count() * 4)

        del dataset

        torch.cuda.empty_cache()

        loss = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss = loss.to(device)

        lr = 0.001
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        shed = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.15)

        is_ = [0, sub_epoches_num - 1]  # printing accuracy

        for i in range(sub_epoches_num):
            print("SubEpoch {}".format(i))
            train(model, loader, loss, optimizer, device, shed)
            if i in is_:
                test(model, loader, device)

        del loader

        torch.cuda.empty_cache()

        torch.save(model.state_dict(), path.format('v', number + 1))

        print("Finish epoch {}\n".format(number))


main(33, 15)
