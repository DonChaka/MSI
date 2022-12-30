# class for consequent in fuzzy systems
from antecedent import Antecedent


class Consequent(Antecedent):

    def set_mf(self, name: str, mf: str, values: list):
        assert mf in ['trimf', 'trapmf', 'constant']

        if mf == 'constant':
            assert len(values) == 1
            self.membership_functions[name] = lambda x: values[0] * x
        elif mf == 'trimf':
            assert len(values) == 3
            self.membership_functions[name] = lambda x: values[1] * x
        elif mf == 'trapmf':
            assert len(values) == 4
            if values[0] == values[1]:
                self.membership_functions[name] = lambda x: values[2] * x
            else:
                self.membership_functions[name] = lambda x: values[1] * x
