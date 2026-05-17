from functools import partial
from .eda import EDA

t = EDA()


def random_deletion(p):
    return partial(t.random_deletion, p=p)
