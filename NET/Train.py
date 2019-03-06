from Net import *
import torch


def train(model, train_loader, criterion1, criterion2, alpha, optimizer, scheduler):
    model.train()

    scheduler.step()

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

        output1, output2 = model(data)
        target1, target2 = target

        loss1 = criterion1(output1, target1)
        loss2 = criterion2(output2, target2)
        loss = loss1 + alpha * loss2

        loss.backward()
        optimizer.step()

    print("Training finished\n")
