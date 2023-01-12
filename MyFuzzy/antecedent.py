import matplotlib.pyplot as plt


class Antecedent:
    def __init__(self, values, name):
        self.values = values
        self.name = name
        self.membership_functions = {}

    def __getitem__(self, item):
        return self.membership_functions[item]

    def __missing__(self, key):
        self.membership_functions[key] = (self.name, lambda x: 0)
        return self.membership_functions[key]

    def auto_trimf(self):
        a, b, c = self.values[0], (self.values[0] + self.values[-1]) / 2, self.values[-1]
        self.set_mf('low', 'trimf', [a, a, b])
        self.set_mf('medium', 'trimf', [a, b, c])
        self.set_mf('high', 'trimf', [b, c, c])

    def set_mf(self, name: str, mf: str, values: list):
        assert mf in ['trimf', 'trapmf']

        if mf == 'trimf':
            assert len(values) == 3
            a, b, c = values
            assert a <= b <= c

            self.membership_functions[name] = (self.name, lambda x: 1 if a == b and b - a == 0 and a <= x <= b else
                min(max((x - a) / (b - a), 0), 1) if a <= x <= b else
                0 if b == c and c - b == 0 and b <= x <= c else
                min(max((c - x) / (c - b), 0), 1) if b <= x <= c else 0
            )

        elif mf == 'trapmf':
            assert len(values) == 4
            a, b, c, d = values
            assert a <= b <= c <= d

            self.membership_functions[name] = (self.name, lambda x: 1 if a == b and b - a == 0 and a <= x <= b else
                min(max((x - a) / (b - a), 0), 1) if a <= x <= b else
                1 if b <= x <= c else
                0 if c == d and d - c == 0 and c <= x <= d else
                min(max((d - x) / (d - c), 0), 1) if c <= x <= d else 0
            )

    def view(self):
        fig, ax = plt.subplots()
        for name, mf in self.membership_functions.items():
            ax.plot(self.values, [mf[1](x) for x in self.values], label=name)

        ax.legend()
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        plt.xlabel(self.name)
        plt.ylabel('Membership')
        plt.xlim(min(self.values), max(self.values))
        plt.ylim(0, 1.01)

        plt.show()


if __name__ == '__main__':
    import numpy as np

    values = np.arange(0, 10.1, 1)
    test = Antecedent(values, 'test')
    # test.set_mf('low', 'trimf', [0, 0, 5])
    # test.set_mf('medium', 'trimf', [0, 5, 10])
    # test.set_mf('high', 'trimf', [5, 10, 10])

    # test.set_mf('low', 'trapmf', [0, 0, 2, 4])
    # test.set_mf('medium', 'trapmf', [2, 4, 6, 8])
    # test.set_mf('high', 'trapmf', [6, 8, 10, 10])

    test.auto_trimf()

    test.view()
