from Dataset import *
from Net import *
import torch
import torch.nn
import torch.optim
import torch.cuda
from torch.utils.data import DataLoader
import time
import os


def train(model, train_loader, criterion1, optimizer):
    start = time.clock()

    model.train()

    batches = len(train_loader)
    percent = {int(batches * 1 / 5): 20,
               int(batches * 2 / 5): 40,
               int(batches * 3 / 5): 60,
               int(batches * 4 / 5): 80,
               batches - 1: 100}

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx in percent:
            finish = time.clock()
            print("{}% ready, time = {}".format(percent[batch_idx], finish - start))

        optimizer.zero_grad()

        data = data.type(torch.cuda.FloatTensor)

        target1, target2 = target

        target1, target2 = target1.to(device), target2.to(device)

        output1, _ = model(data)

        loss = criterion1(output1, target2)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Training finished, total time = {}\n".format(finish - start))


print("Available cudas {}\n".format(torch.cuda.device_count()))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device, '\n')

model_ = Net()
if torch.cuda.device_count() > 1:
    model_ = torch.nn.DataParallel(model_)
    print("Using {} cudas for our model\n".format(torch.cuda.device_count()))

path = "model{}.pth"

for number in range(10):
    print("Start epoch {}".format(number))

    if os.path.isfile(path.format(number)):
        model_.load_state_dict(torch.load(path.format(number), map_location=lambda storage, loc: storage))

    model_ = model_.to(device)

    print("Model ready\n")

    start = time.clock()

    dataset = make_dataset(1000 + 50000 * number, 30000 + 50000 * number)

    print("Dataset ready, time {}\n".format(time.clock() - start))

    start = time.clock()

    loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True,
                        num_workers=torch.cuda.device_count() * 4, drop_last=True)

    print("DataLoader ready, time {}\n".format(time.clock() - start))

    loss1 = torch.nn.L1Loss()
    lr = 0.002
    optimizer = torch.optim.Adam(model_.parameters(), lr=lr)

    for i in range(3):
        print("SubEpoch {}\n".format(i))
        train(model_, loader, loss1, optimizer)

    torch.save(model_.state_dict(), path.format(number + 1))

    print("Finish epoch {}\n".format(number))
