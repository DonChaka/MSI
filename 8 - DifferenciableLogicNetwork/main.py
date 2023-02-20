import torch
from LogicLayer import LogicLayer
from OutputLayer import GroupOutput
from LogicNetwork import LogicNetwork
from torch import nn
from torch.nn import Flatten
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

transform = transforms.Compose([transforms.ToTensor()])

train_loader = DataLoader(datasets.MNIST(root='data', train=True, download=True, transform=transform), batch_size=32, shuffle=True)
test_loader = DataLoader(datasets.MNIST(root='data', train=False, download=True, transform=transform), batch_size=1, shuffle=True)

model = LogicNetwork([
    Flatten(),
    LogicLayer(784, 2048),
    LogicLayer(2048, 2048),
    LogicLayer(2048, 2048),
    LogicLayer(2048, 2048),
    LogicLayer(2048, 2048),
    LogicLayer(2048, 2048),
    LogicLayer(2048, 2048),
    GroupOutput(10)
])

model.fit(epochs=10, train_loader=train_loader, verbose=2)

preds = []
for x, y in test_loader:
    preds.append(model(x).argmax().item())

print(classification_report([y.item() for x, y in test_loader], preds))
