import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from mytorch import MyConv2D, MyMaxPool2D
from torch.utils.data import DataLoader


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Net(nn.Module):
    def __init__(self):

        """
        My custom network
        [hint]
        * See the instruction PDF for details
        * Only allow to use MyConv2D and MyMaxPool2D
        * Set the bias argument to True
        """
        super().__init__()
        
        ## Define all the layers
        ## Use MyConv2D, MyMaxPool2D for the network
        # ----- TODO -----

        self.conv_first = MyConv2D(1, 3, (3, 3), 1, 1)
        self.act_first = nn.ReLU()
        self.pool_first = MyMaxPool2D((2, 2), (2, 2))
        self.conv_second = MyConv2D(3, 6, (3, 3), 1, 1)
        self.act_second = nn.ReLU()
        self.pool_second = MyMaxPool2D((2, 2), (2, 2))
        self.flatten_layer = nn.Flatten()
        self.lin_first = nn.Linear(294, 128)
        self.act_third = nn.ReLU()
        self.lin_second = nn.Linear(128, 10)



    def forward(self, x):
        
        # ----- TODO -----

        x = self.conv_first(x)
        x = self.act_first(x)
        x = self.pool_first(x)
        x = self.conv_second(x)
        x = self.act_second(x)
        x = self.pool_second(x)
        x = self.flatten_layer(x)
        x = self.lin_first(x)
        x = self.act_third(x)
        x = self.lin_second

        return x


if __name__ == "__main__":

    # set param
    setup_seed(18786)
    batch_size = 128
    num_epoch = 5
    lr = 1e-4

    ## Load dataset

    # ----- TODO -----

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True)

    print(f"LOAD DATASET: TRAIN {len(trainset)} | TEST: {len(valset)}")

    ## Load my neural network
    # ----- TODO -----

    model = Net()
    

    ## Define the criterion and optimizer
    # ----- TODO -----

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)

    ## Training and evaluation
    ## Feel free to record the loss and accuracy numbers
    ## Hint: you could separate the training and evaluation
    ## process into 2 different functions for each epoch

    t_loss = []
    t_acc = []
    v_loss = []
    v_acc = []
    for epoch in range(num_epoch): 
        # ----- TODO -----
        
        train_loss = train_count = 0
        train_correct = train_total = 0
        val_loss = val_count = 0
        val_correct = val_total = 0
        train_accuracy = val_accuracy = 0
        for data, target in trainloader:
            optimizer.zero_grad()
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_count += 1
            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += data.size(0)

        train_loss /= train_count
        train_accuracy = 100. * train_correct / train_total
        t_loss.append(train_loss)
        t_acc.append(train_accuracy)

        for data, target in trainloader:
            data = data.view(data.size(0), -1)
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()
            val_count += 1
            pred = output.argmax(dim=1)
            val_correct += (pred == target).sum().item()
            val_total += data.size(0)

        val_loss /= val_count
        val_accuracy = 100. * val_correct / val_total
        v_loss.append(val_loss)
        v_acc.append(val_accuracy)

        print(f'Epoch: {epoch}, Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}, Validation loss: {val_loss:.4f}, Validation accuracy: {val_accuracy:.4f}')


    ## Plot the loss and accuracy curves
    # ----- TODO -----

    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(10, 5), dpi=120)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, t_loss, label = 'Training loss')
    plt.plot(epochs, v_loss, label = 'Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Losses')
    plt.legend()
    plt.title('Loss vs epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, t_acc, label = 'Training accuracy')
    plt.plot(epochs, v_acc, label = 'Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs epochs')

    plt.tight_layout()
    plt.show()
