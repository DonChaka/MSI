from torch import Tensor
from torch.nn import Module


class GroupOutput(Module):
    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.n_classes = n_classes

    def forward(self, x: Tensor) -> Tensor:
        return x.reshape(*x.shape[:-1], self.n_classes, x.shape[-1] // self.n_classes).sum(-1) / self.n_classes
