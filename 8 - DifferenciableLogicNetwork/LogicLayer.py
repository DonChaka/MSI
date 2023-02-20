import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Module
from torch.nn.functional import one_hot, softmax


class LogicLayer(Module):

    @staticmethod
    def bin_op(a, b, i):
        assert a[0].shape == b[0].shape, (a[0].shape, b[0].shape)
        if a.shape[0] > 1:
            assert a[1].shape == b[1].shape, (a[1].shape, b[1].shape)

        if i == 0:
            return torch.zeros_like(a)
        elif i == 1:
            return a * b
        elif i == 2:
            return a - a * b
        elif i == 3:
            return a
        elif i == 4:
            return b - a * b
        elif i == 5:
            return b
        elif i == 6:
            return a + b - 2 * a * b
        elif i == 7:
            return a + b - a * b
        elif i == 8:
            return 1 - (a + b - a * b)
        elif i == 9:
            return 1 - (a + b - 2 * a * b)
        elif i == 10:
            return 1 - b
        elif i == 11:
            return 1 - b + a * b
        elif i == 12:
            return 1 - a
        elif i == 13:
            return 1 - a + a * b
        elif i == 14:
            return 1 - a * b
        elif i == 15:
            return torch.ones_like(a)

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        self.weights = nn.parameter.Parameter(torch.randn(size=(out_dim, 16)))

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.idx = self.get_indexes()

    def bin_forward(self, a: Tensor, b: Tensor, indices: Tensor) -> Tensor:
        res = torch.zeros_like(a)

        for i in range(16):
            res += indices[:, i] * self.bin_op(a, b, i)

        return res

    def get_indexes(self) -> tuple:
        data = torch.randperm(self.in_dim)[torch.randperm(2 * self.out_dim) % self.in_dim]
        data = data.reshape(int(len(data) / self.out_dim), self.out_dim)

        return data[0], data[1]

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.bin_forward(
                a=x[:, self.idx[0]],
                b=x[:, self.idx[1]],
                indices=softmax(self.weights, dim=-1)
            )

        return self.bin_forward(
            a=x[:, self.idx[0]],
            b=x[:, self.idx[1]],
            indices=one_hot(self.weights.argmax(-1), 16)
        )