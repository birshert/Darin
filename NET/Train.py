from Dataset import *
from Net import *
import torch
import torch.nn
import torch.optim
import torch.cuda
from torch.utils.data import DataLoader
import time
import os
import gc


def train(model, train_loader, criterion1, optimizer, device, shed):
    start = time.clock()

    model.train()
    shed.step()

    batches = len(train_loader)
    percent = {int(batches * 1 / 5): 20,
               int(batches * 2 / 5): 40,
               int(batches * 3 / 5): 60,
               int(batches * 4 / 5): 80,
               batches - 1: 100}

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx in percent:
            print("{}% ready".format(percent[batch_idx]))

        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.type(torch.cuda.FloatTensor)
        else:
            data = data.type(torch.FloatTensor)

        target = target.to(device)

        output1 = model(data)

        loss = criterion1(output1, target)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Training finished, total time = {}\n".format(finish - start))


def test(model, test_loader, device):
    model.eval()

    with torch.no_grad():
        correct = 0

        for data, target in test_loader:
            if torch.cuda.is_available():
                data = data.type(torch.cuda.FloatTensor)
            else:
                data = data.type(torch.FloatTensor)

            output = model(data)

            target = target.to(device)

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()

        print('Accuracy: {}/{} ({:.3f}%)\n'.format(correct, len(test_loader.dataset),
                                                   100. * correct / len(test_loader.dataset)))


def main(epoches_num, sub_epoches_num):
    print("Available cudas {}\n".format(torch.cuda.device_count()))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device, '\n')

    model_ = Net()
    if torch.cuda.device_count() > 1:
        model_ = torch.nn.DataParallel(model_)
        print("Using {} cudas for our model\n".format(torch.cuda.device_count()))

    path = "model{}.pth"

    for number in range(epoches_num):
        print("Start epoch {}".format(number))

        if os.path.isfile(path.format(number)):
            model_.load_state_dict(torch.load(path.format(number), map_location=lambda storage, loc: storage))

        model_ = model_.to(device)

        print("Model ready\n")

        start = time.clock()

        dataset = make_dataset(10 * number, 100000 * (number + 1))

        print("Dataset ready, size {}, time {}\n".format(len(dataset), time.clock() - start))

        start = time.clock()

        loader = DataLoader(dataset, batch_size=2048, shuffle=True, pin_memory=True,
                            num_workers=torch.cuda.device_count() * 4, drop_last=False)

        print("DataLoader ready, time {}\n".format(time.clock() - start))

        del dataset

        loss1 = torch.nn.CrossEntropyLoss()
        if torch.cuda.is_available():
            loss1 = loss1.to(device)

        lr = 0.1
        optimizer = torch.optim.Adam(model_.parameters(), lr=lr)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

        for i in range(sub_epoches_num):
            print("SubEpoch {}\n".format(i))
            train(model_, loader, loss1, optimizer, device, exp_lr_scheduler)
            torch.save(model_.state_dict(), path.format(i + 1))
            test(model_, loader, device)

        del loader

        gc.collect()

        torch.save(model_.state_dict(), 'total' + path.format(number))

        print("Finish epoch {}\n".format(number))


main(3, 20)
