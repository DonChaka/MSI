import torch
from tqdm import tqdm

from LogicLayer import LogicLayer
from OutputLayer import GroupOutput
from LogicNetwork import LogicNetwork
from torch import nn
from torch.nn import Flatten
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                ])

train_loader = DataLoader(datasets.MNIST(root='data', train=True, download=True, transform=transform), batch_size=64,
                          shuffle=True)
test_loader = DataLoader(datasets.MNIST(root='data', train=False, download=True, transform=transform), batch_size=1,
                         shuffle=False)

model = LogicNetwork([
    Flatten(),
    LogicLayer(784, 1000),
    LogicLayer(1000, 1000),
    LogicLayer(1000, 1000),
    LogicLayer(1000, 1000),
    GroupOutput(10)
])

model.fit(epochs=2, train_loader=train_loader, verbose=2)

preds = []
true_y = []
for x, y in tqdm(test_loader, desc='Testing'):
    preds.append(model(x).argmax().item())
    true_y.append(y.item())

print(classification_report(true_y, preds))
