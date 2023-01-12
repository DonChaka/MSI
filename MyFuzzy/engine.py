import numpy as np

from rule import Rule


class FuzzyEngine:
    def __init__(self, rules: list[Rule]):
        self.rules = rules
        self.input = {}
        self.result = None

    def __setitem__(self, key, value):
        self.input[key] = value

    def compute(self):
        weights = []
        vals = []

        for rule in self.rules:
            weights.append(rule(self.input))
            vals.append(rule.consequent)

        self.result = np.average(a=vals, weights=weights)

        return self.result
