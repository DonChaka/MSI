import numpy as np


class Rule:
    _OPERATIONS = {'and': lambda x: np.min(x), 'or': lambda x: np.max(x)}

    def __init__(self, antecedents: list, operations: list[str], consequent: int):
        for op in operations:
            if op not in self._OPERATIONS.keys():
                raise ValueError(f"Operator {op} is not supported")

        self.antecedents = antecedents
        self.operations = operations
        self.consequent = consequent

    def __call__(self, args: dict):
        if not len(self.operations):
            return 0

        ant_vals = []
        try:
            for ant in self.antecedents:
                ant_vals.append(ant[1](args[ant[0]]))
        except KeyError:
            raise KeyError(f'Missing value for antecedent {ant[0]}')

        oped_vals = [self._OPERATIONS[op](ant_vals) for op in self.operations]
        return np.average(oped_vals)