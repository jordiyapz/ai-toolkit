import enum
from random import random as rand


def translate(value, leftMin, leftMax, rightMin, rightMax):
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)


def toDecimal(bin_list):
    dec = 0
    l = len(bin_list) - 1
    for i, x in enumerate(bin_list):
        dec += pow(2, l - i) * x
    return dec


# %% enumerasi
class Stop(enum.Enum):
    MAX_IT = 0      # always on, stop when max iteration reached
    TRESHOLD = 1    # stop if fitness >= treshold value
    NO_IMPROVE = 3  # stop if no improvement for certain generation


# %% Gea
class Individu:
    def __init__(self,
                 ranges: tuple = ((0, 1), (0, 1)),
                 resolusi: int = 5,
                 kromosom: list = None):

        assert type(ranges) is tuple
        for rg in ranges:
            assert len(rg) is 2

        self.kromosom = kromosom or \
            [round(rand()) for _ in range(resolusi * len(ranges))]
        self.res = resolusi
        self.ranges = ranges

    def getFenotip(self):
        l = self.res
        up = 2**l-1
        return tuple(translate(toDecimal(self.kromosom[l*i:l*(i+1)]),
                               0, up, ran[0], ran[1])
                     for i, ran in enumerate(self.ranges))
