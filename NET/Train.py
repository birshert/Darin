from Net import *
from Dataset import *
import torch.nn
from torch.utils.data import DataLoader
import torch.optim


def train(model, train_loader, criterion1, criterion2, alpha, optimizer):
    model.train()

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

        output1, _ = model(data)
        target2 = target[1]
        target2 = target2.type(torch.FloatTensor)

        loss = loss1(output1, target2)

        loss.backward()
        optimizer.step()

    print("Training finished\n")


data = make_dataset(10000)
loader = DataLoader(data, 64, True)

print("LOADER READY, MY LORD\n\n")

path = "model1.1.pth"

model = Net()
model.load_state_dict(torch.load(path))

optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

loss1 = torch.nn.MSELoss()
loss2 = torch.nn.MSELoss()

l = 0.1

train(model, loader, loss1, loss2, l, optimizer)

torch.save(model.state_dict(), path)
