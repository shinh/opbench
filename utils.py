import chainer
import cupy


def as_list(a):
    if isinstance(a, tuple):
        return list(a)
    else:
        return [a]


def to_gpu(arrays):
    return [cupy.array(a) for a in arrays]


def to_cpu(arrays):
    out = []
    for a in arrays:
        if isinstance(a, chainer.Variable):
            a = a.array
        out.append(chainer.cuda.to_cpu(a))
    return out
