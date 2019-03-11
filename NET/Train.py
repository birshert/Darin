from Dataset import *
from Net import *
import torch
import torch.nn
import torch.optim
import torch.cuda
from torch.utils.data import DataLoader
import time


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

        output1, _ = model(data)
        target_ = target[1]
        target_ = target_.type(torch.FloatTensor)

        loss = criterion1(output1, target_)

        loss.backward()
        optimizer.step()

    finish = time.clock()
    print("Training finished, total time = {}\n".format(finish - start))


print("Available cudas {}\n".format(torch.cuda.device_count()))

id_ = 0
path = "model{}.pth"
model_ = Net()
omega_lul_net = torch.nn.DataParallel(model_)  # default all devices
# omega_lul_net.load_state_dict(torch.load(path.format(id_)))
print("Model ready\n")

optim = torch.optim.Adam(omega_lul_net.parameters())

loss1 = torch.nn.L1Loss()

dataset = make_dataset(1000, 3000)
loader = DataLoader(dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=torch.cuda.device_count())

print("Dataset ready\n")

train(omega_lul_net, loader, loss1, optim)
torch.save(omega_lul_net.state_dict(), path.format(id_ + 1))
