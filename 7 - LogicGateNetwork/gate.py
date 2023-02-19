from numpy import ndarray, logical_or, logical_and, logical_not, logical_xor


class Gate:
    @staticmethod
    def And(x: ndarray) -> ndarray:
        return logical_and(*x.T).reshape(-1, 1)

    @staticmethod
    def Or(x: ndarray) -> ndarray:
        return logical_or(*x.T).reshape(-1, 1)

    @staticmethod
    def Xor(x: ndarray) -> ndarray:
        return logical_xor(*x.T).reshape(-1, 1)

    @staticmethod
    def Nand(x: ndarray) -> ndarray:
        return logical_not(Gate.And(x))


