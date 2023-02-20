import numpy as np
from torch.nn import CrossEntropyLoss, Sequential
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch import no_grad
from tqdm import tqdm


class LogicNetwork:
    def __init__(self, layers: list, loss=None, optimizer=None) -> None:
        self.layers = layers

        self.model = Sequential()

        for layer in self.layers:
            self.model.append(layer)

        self.loss = loss if loss else CrossEntropyLoss()
        self.optimizer = optimizer if optimizer else Adam(params=self.model.parameters(), lr=0.01)

    def fit(self, epochs: int, train_loader: DataLoader, verbose: int = 0) -> None:
        train_state = self.model.training
        self.model.training=True

        epoch_tqdm = tqdm(range(epochs), disable=(verbose != 1))
        for epoch in epoch_tqdm:
            step_tqdm = tqdm(train_loader, disable=(verbose != 2), desc=f'Epoch: {epoch}/{epochs}')
            for x, y in step_tqdm:
                preds = self.model(x)
                loss = self.loss(preds, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self.model.training = train_state

    def __call__(self, x):
        with no_grad():
            return self.model(x)
