import numpy as np
from numpy import ndarray, logical_or, logical_and, logical_not, logical_xor
from typing import Callable, List

BIN2GATE = {
    '00': lambda x: logical_and(*x.T).reshape(-1, 1),
    '01': lambda x: logical_or(*x.T).reshape(-1, 1),
    '10': lambda x: logical_xor(*x.T).reshape(-1, 1),
    '11': lambda x: logical_not(logical_and(*x.T).reshape(-1, 1))
}

NEURON_LENGTH = 22

class GateNeuron:
    def __init__(self, indexes: np.ndarray, gate: Callable) -> None:
        self.indexes = indexes
        self.gate = gate

    def set_params(self, indexes: np.ndarray, gate: Callable) -> None:
        self.indexes = indexes
        self.gate = gate

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.gate(x[:, self.indexes])


class Layer:
    def __init__(self, n_features: int, units: int) -> None:
        self.n_features = n_features
        self.units = units
        self.neurons = []

        self.init_neurons()

    def init_neurons(self):
        for _ in range(self.units):
            self.neurons.append(GateNeuron(
                indexes=np.random.choice(a=self.n_features, size=2, replace=False),
                gate=np.random.choice(list(BIN2GATE.values()))
            ))

    def set_params(self, gates: list, indexes: list) -> None:
        for neuron, gate, idx in zip(self.neurons, gates, indexes):
            neuron.set_params(indexes=np.clip(idx, 0, self.n_features - 1), gate=gate)

    def forward(self, x: np.ndarray) -> np.ndarray:
        result = [self.neurons[i].forward(x) for i in range(self.units)]

        return np.array(result).reshape(x.shape[0], self.units)

    def get_config_length(self) -> int:
        return self.units * NEURON_LENGTH


class LogicGateNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def get_feature_vector_length(self) -> int:
        ret = 0
        for layer in self.layers:
            ret += layer.get_config_length()

        return ret

    def set_params(self, config_vector: str) -> None:
        i = 0
        for layer in self.layers:
            l_conf = config_vector[i:i + layer.get_config_length()]
            gates = [BIN2GATE[l_conf[j:j + 2]] for j in range(0, len(l_conf), NEURON_LENGTH)]
            indexes = [(int(l_conf[j+2:j + 12], 2), int(l_conf[j+12: j+22])) for j in range(0, len(l_conf), NEURON_LENGTH)]
            layer.set_params(gates=gates, indexes=indexes)
            i += layer.get_config_length()
