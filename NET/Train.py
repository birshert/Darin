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


def train(model, train_loader, criterion1, optimizer, device):
    start = time.clock()

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()

        if torch.cuda.is_available():
            data = data.type(torch.cuda.FloatTensor)
        else:
            data = data.type(torch.FloatTensor)

        target1, target2 = target

        target2 = target2.to(device)

        output1, _ = model(data)

        loss = criterion1(output1, target2)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Training finished, total time = {}\n".format(finish - start))


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

        dataset = make_dataset(4096 * number, 4096 * (number + 1))

        print("Dataset ready, time {}\n".format(time.clock() - start))

        start = time.clock()

        loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True,
                            num_workers=torch.cuda.device_count() * 4, drop_last=True)

        print("DataLoader ready, time {}\n".format(time.clock() - start))

        del dataset

        loss1 = torch.nn.L1Loss()
        if torch.cuda.is_available():
            loss1 = loss1.to(device)

        lr = 0.1
        optimizer = torch.optim.Adam(model_.parameters(), lr=lr)

        for i in range(sub_epoches_num):
            print("SubEpoch {}\n".format(i))
            train(model_, loader, loss1, optimizer, device)

        del loader

        gc.collect()

        torch.save(model_.state_dict(), path.format(number + 1))

        print("Finish epoch {}\n".format(number))


main(1000, 5)
