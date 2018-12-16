import chainer

import driver
import utils


class ChainerDriver(driver.Driver):
    def name(self):
        return 'chainer'

    def run_first(self, task, inputs, sample_outputs):
        self.model = task.model
        self.model.to_gpu()
        self.inputs = utils.to_gpu(inputs)
        gpu_outputs = self.run_task()
        gpu_outputs = utils.as_list(gpu_outputs)
        outputs = utils.to_cpu(gpu_outputs)
        return outputs

    def run_task(self):
        outputs = self.model(*self.inputs)
        return outputs


def get_driver():
    return ChainerDriver()
