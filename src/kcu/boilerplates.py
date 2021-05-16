import torch
import matplotlib.pyplot as plt
import numpy as np


def train_classifier(
    model, optimizer, train_loader, device, epochs, loss_fct, val_loader, show_plot
):
    """ TRAINING """
    test_history_acc = []
    train_history_loss = []
    test_history_loss = []
    train_loss = 0
    for epoch in range(epochs):
        model.train()
        for batch, (X, y) in enumerate(train_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fct(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # if batch % 10 == 0:
            #    loss, current = loss.item(), batch * len(X)
            #    print(f"loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}]")

        train_loss = train_loss / len(train_loader.dataset)
        train_history_loss.append(train_loss)

        """ EVALUATION """
        model.eval()
        test_loss, correct = 0, 0
        for batch, (X, y) in enumerate(val_loader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fct(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss /= len(val_loader.dataset)
        correct /= len(val_loader.dataset)
        test_history_loss.append(test_loss)
        test_history_acc.append(correct)

        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )
        
        if show_plot:
            plt.plot(np.arange(epoch + 1), test_history_acc)
            plt.plot(np.arange(epoch + 1), train_history_loss)
            plt.plot(np.arange(epoch + 1), test_history_loss)
            plt.savefig("results.png")
            plt.show()
