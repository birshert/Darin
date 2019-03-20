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


def train(model, loader, criterion1, criterion2, optimizer, device, shed):
    start = time.clock()

    model.train()
    shed.step()

    for data, target in loader:
        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.type(torch.cuda.FloatTensor)
        else:
            data = data.type(torch.FloatTensor)

        target1 = torch.stack([target[i][0] for i in range(len(target))]).type(torch.LongTensor)
        target2 = torch.stack([target[i][1] for i in range(len(target))])

        target1 = target1.to(device)

        target2 = target2.to(device)

        policy, value = model(data)

        loss1 = criterion1(policy, target1)
        loss2 = criterion2(value, target2)

        loss = loss1 * 0.5 + loss2 * 0.5

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Training finished, total time = {}".format(finish - start))


def test(model, loader, device):
    model.eval()

    with torch.no_grad():
        correct_p = 0
        correct_v = 0

        for data, target in loader:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            policy, value = model(data)

            policy = F.softmax(policy, dim=1)

            target1 = torch.stack([target[i][0] for i in range(len(target))]).type(torch.LongTensor)
            target2 = torch.stack([target[i][1] for i in range(len(target))])

            target1 = target1.to(device)

            target2 = target2.to(device)

            pred = policy.data.max(1, keepdim=True)[1]
            correct_p += int(pred.eq(target1.data.view_as(pred)).sum())

            for i in range(target2.size(0)):
                if target2[i] * value[i] > 0:
                    correct_v += 1

        print("Test: p {}/{}, v {}/{}".format(correct_p, len(loader.dataset), correct_v, len(loader.dataset)))


def main(epoches_num, sub_epoches_num):
    gc.enable()
    print("Available cudas {}\n".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device, '\n')

    model = Net()

    if torch.cuda.device_count() > 1:
        modelp = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
        print("Using {} cudas for our model\n".format(torch.cuda.device_count()))

    path = "model_{}.pth"

    for number in range(epoches_num):
        print("Start epoch {}".format(number))

        if os.path.isfile(path.format(number)):
            model.load_state_dict(torch.load(path.format(number), map_location=lambda storage, loc: storage))
            print("Model loaded\n")

        model = model.to(device)

        start = time.clock()

        count = 10000

        torch.cuda.empty_cache()

        dataset = make_dataset(count * number, count * (number + 1))

        print('Dataset ready, size {}, time {}'.format(len(dataset), time.clock() - start))

        loader = DataLoader(dataset, batch_size=4096, shuffle=True, drop_last=False, pin_memory=True, num_workers=4)

        del dataset

        torch.cuda.empty_cache()

        loss1 = torch.nn.CrossEntropyLoss()
        loss2 = torch.nn.MSELoss()

        loss1 = loss1.to(device)
        loss2 = loss2.to(device)

        lr = 0.01

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        shed = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.2)

        test(model, loader, device)

        for i in range(sub_epoches_num):
            print("SubEpoch {}".format(i))
            train(model, loader, loss1, loss2, optimizer, device, shed)

        test(model, loader, device)

        del loader

        torch.cuda.empty_cache()

        torch.save(model.state_dict(), path.format(number + 1))

        print("Finish epoch {}\n".format(number))


main(150, 20)
