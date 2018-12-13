import chainer

import driver
import utils


class ChainerDriver(driver.Driver):
    def name(self):
        return 'chainer'

    def run_first(self, task, inputs, sample_outputs):
        task.model.to_gpu()
        self.inputs = utils.to_gpu(inputs)
        gpu_outputs = self.run_task(task)
        gpu_outputs = utils.as_list(gpu_outputs)
        outputs = utils.to_cpu(gpu_outputs)
        return outputs

    def run_task(self, task):
        outputs = task.model(*self.inputs)
        chainer.cuda.Stream.null.synchronize()
        return outputs


def get_driver():
    return ChainerDriver()
