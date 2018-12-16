import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np


class Layer(object):
    def __init__(self, bsize, ichan, ochan, wh, ksize, stride, pad, postprocs):
        self.bsize = bsize
        self.ichan = ichan
        self.ochan = ochan
        self.wh = wh
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
        self.postprocs = postprocs
        owh = (wh + pad * 2 - ksize + stride) / stride
        self.flops = bsize * ichan * ochan * owh * owh * ksize * ksize


BatchNorm = 1
Relu = 2
MaxPool = 3
AvgPool = 4

def resnet50_layers(bsize):
    return [
        Layer(bsize, 3, 64, 224, 7, 2, 3, [BatchNorm,Relu,MaxPool]),
        Layer(bsize, 64, 64, 56, 1, 1, 0, [BatchNorm]),
        Layer(bsize, 64, 256, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 64, 64, 56, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 64, 256, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 64, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 64, 64, 56, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 64, 256, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 64, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 64, 64, 56, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 64, 256, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 512, 56, 1, 2, 0, [BatchNorm]),
        Layer(bsize, 256, 128, 56, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 128, 128, 56, 3, 2, 1, [BatchNorm,Relu]),
        Layer(bsize, 128, 512, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 128, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 128, 128, 28, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 128, 512, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 128, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 128, 128, 28, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 128, 512, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 128, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 128, 128, 28, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 128, 512, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 1024, 28, 1, 2, 0, [BatchNorm]),
        Layer(bsize, 512, 256, 28, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 256, 28, 3, 2, 1, [BatchNorm,Relu]),
        Layer(bsize, 256, 1024, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 1024, 256, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 256, 14, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 256, 1024, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 1024, 256, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 256, 14, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 256, 1024, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 1024, 256, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 256, 14, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 256, 1024, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 1024, 256, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 256, 14, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 256, 1024, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 1024, 256, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 256, 256, 14, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 256, 1024, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 1024, 2048, 14, 1, 2, 0, [BatchNorm]),
        Layer(bsize, 1024, 512, 14, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 512, 14, 3, 2, 1, [BatchNorm,Relu]),
        Layer(bsize, 512, 2048, 7, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 2048, 512, 7, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 512, 7, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 512, 2048, 7, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 2048, 512, 7, 1, 1, 0, [BatchNorm,Relu]),
        Layer(bsize, 512, 512, 7, 3, 1, 1, [BatchNorm,Relu]),
        Layer(bsize, 512, 2048, 7, 1, 1, 0, [BatchNorm,Relu,AvgPool]),
    ]


class Conv(chainer.Chain):
    def __init__(self, name, bsize, index, layer):
        super(Conv, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(layer.ichan,
                                        layer.ochan,
                                        layer.ksize,
                                        layer.stride,
                                        layer.pad,
                                        nobias=True)
        self.category = '%s_conv_bs%d' % (name, bsize)
        self.name = '%s_%02d' % (self.category, index)
        self.layer = layer
        self.flops = layer.flops

    def forward(self, x):
        return self.conv(x)

    def inputs(self):
        bsize = self.layer.bsize
        ichan = self.layer.ichan
        wh = self.layer.wh
        return np.random.normal(size=(bsize, ichan, wh, wh)).astype(np.float32)

    def info(self):
        l = self.layer
        return [
            ('flops', l.flops),
            ('bsize', l.bsize),
            ('ichan', l.ichan),
            ('ochan', l.ochan),
            ('wh', l.wh),
            ('ksize', l.ksize),
            ('stride', l.stride),
            ('pad', l.pad),
        ]

class ConvFuse(chainer.Chain):
    def __init__(self, name, bsize, index, layer):
        super(ConvFuse, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(layer.ichan,
                                        layer.ochan,
                                        layer.ksize,
                                        layer.stride,
                                        layer.pad,
                                        nobias=True)
            if BatchNorm in layer.postprocs:
                self.bn = L.BatchNormalization(layer.ochan)

        postprocs = []
        for pp in layer.postprocs:
            if pp == MaxPool or pp == AvgPool:
                continue
            postprocs.append({
                BatchNorm: 'bn',
                Relu: 'relu',
            }[pp])

        self.category = '%s_convfuse_bs%d' % (name, bsize)
        self.name = '%s_%d' % (self.category, index)
        self.layer = layer
        self.flops = layer.flops

    def forward(self, x):
        y = self.conv(x)
        for pp in self.layer.postprocs:
            if pp == MaxPool or pp == AvgPool:
                continue
            y = {
                BatchNorm: self.bn,
                Relu: F.relu,
            }[pp](y)
        return y

    def inputs(self):
        bsize = self.layer.bsize
        ichan = self.layer.ichan
        wh = self.layer.wh
        return np.random.normal(size=(bsize, ichan, wh, wh)).astype(np.float32)

    def info(self):
        l = self.layer
        bn = int(BatchNorm in l.postprocs)
        relu = int(Relu in l.postprocs)
        return [
            ('flops', l.flops),
            ('bsize', l.bsize),
            ('ichan', l.ichan),
            ('ochan', l.ochan),
            ('wh', l.wh),
            ('ksize', l.ksize),
            ('stride', l.stride),
            ('pad', l.pad),
            ('bn', bn),
            ('relu', relu),
        ]


def get_tasks():
    tasks = []
    for bsize in [1, 32]:
        for i, layer in enumerate(resnet50_layers(bsize)):
            tasks.append(Conv('resnet50', bsize, i, layer))
            tasks.append(ConvFuse('resnet50', bsize, i, layer))
    return tasks
