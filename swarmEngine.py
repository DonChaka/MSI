from typing import Callable, Optional
import numpy as np
from numpy import ndarray
from numpy.random import rand, uniform, normal
from tqdm import tqdm

N_UNITS = 100


def f(x: ndarray) -> float:
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


class Unit:
    best_value: float = float("inf")
    best_positions: Optional[ndarray] = None
    exploration = True

    def __init__(self, n_values: int, function: Callable, weight: Optional[float] = None, c1: float = 1, c2: float = 1,
                 min_values: Optional[ndarray] = None, max_values: Optional[ndarray] = None, cost_function_args=None):
        if max_values is not None:
            assert len(max_values) == n_values or max_values is None
        if min_values is not None:
            assert len(min_values) == n_values or min_values is None
        self.n_values = n_values
        self.function = function
        self.cost_function_args = cost_function_args if cost_function_args is not None else ()
        self.positions = normal(1, .15, n_values)
        self.max_values = max_values if max_values is not None else np.array([float('inf')] * n_values)
        self.min_values = min_values if min_values is not None else np.array([-float('inf')] * n_values)
        self.velocities = normal(0, .5, n_values)
        self.validate_positions()
        self.c1 = c1
        self.c2 = c2
        self.weight = weight if weight is not None else uniform(0.5, 1.5)

        self.best_value = float('inf')
        self.update_best()

    def update_best(self) -> None:
        val = self.function(self.positions, *self.cost_function_args)
        if val < self.best_value:
            self.best_value = val
            self.best_positions = self.positions.copy()
            if val < Unit.best_value:
                Unit.best_value = val
                Unit.best_positions = self.positions.copy()

    def update_velocity(self) -> None:
        if not self.exploration:
            r1 = np.random.rand()
            r2 = np.random.rand()
            self.velocities = self.weight * self.velocities + self.c1 * r1 * (self.best_positions - self.positions) + \
                              self.c2 * r2 * (Unit.best_positions - self.positions)

    def update_position(self) -> None:
        self.positions += self.velocities

    def validate_positions(self) -> None:
        high_index = self.positions > self.max_values
        self.positions[high_index] = self.max_values[high_index]
        self.velocities[high_index] = -self.velocities[high_index]

        low_index = self.positions < self.min_values
        self.positions[low_index] = self.min_values[low_index]
        self.velocities[low_index] = -self.velocities[low_index]

    def update(self) -> None:
        self.update_velocity()
        self.update_position()
        self.validate_positions()
        self.update_best()


class Swarm:
    def __init__(self, n_units: int, n_features: int, cost_function: Callable, c1: float = 1, c2: float = 1,
                 min_values: Optional[ndarray] = None, max_values: Optional[ndarray] = None, cost_function_args=None):
        self.n_units = n_units
        self.n_values = n_features
        self.cost_function = cost_function
        self.function_args = cost_function_args
        self.units = [
            Unit(n_features, cost_function, c1=c1, c2=c2, min_values=min_values, max_values=max_values,
                 cost_function_args=cost_function_args) for _ in range(n_units)
        ]

    def update(self) -> None:
        for unit in self.units:
            unit.update()

    def get_best(self) -> ndarray:
        return Unit.best_positions

    def get_best_value(self) -> float:
        return Unit.best_value

    def switch_exploration(self) -> None:
        Unit.exploration = not Unit.exploration

    def run(self, n_iter=10000, exploration_iters=50, verbose=False):
        for i in tqdm(range(n_iter), disable=not verbose):
            if i == exploration_iters:
                self.switch_exploration()
            self.update()
        return self.get_best(), self.get_best_value()


def main():
    swarm = Swarm(N_UNITS, 2, f, min_values=np.array([-4.5, -4.5]), max_values=np.array([4.5, 4.5]))

    print(swarm.run())


if __name__ == '__main__':
    main()
