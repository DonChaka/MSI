from typing import Callable, Optional


# clas for rule in fuzzy systems
class Rule:
    """


    """

    _OPERATIONS = {'and': lambda x, y: min(x, y), 'or': lambda x, y: max(x, y)}

    def __init__(self, first: Callable, operation: Optional[str], second: Optional[Callable], consequent: Callable):
        if operation is not None:
            assert operation in ['and', 'or']
            assert second is not None

        self.first = first
        self.operation = operation
        self.second = second
        self.consequent = consequent

    def __call__(self, *args, **kwargs):
        return self.consequent(self.first(*args, **kwargs) if self.operation is None else
                               self._OPERATIONS[self.operation](self.first(*args, **kwargs), self.second(*args, **kwargs)))
