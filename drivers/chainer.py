import chainer

import driver


class ChainerDriver(driver.Driver):
    def run_task(self, task):
        return task.model(*self.inputs)

    def get_result(self, task, inputs):
        task.model.to_gpu()
        self.inputs = [task.model.xp.array(input) for input in inputs]
        gpu_outputs = self.run_task(task)
        if not isinstance(gpu_outputs, tuple):
            gpu_outputs = [gpu_outputs]
        outputs = [chainer.cuda.to_cpu(v.array) for v in gpu_outputs]
        chainer.cuda.Stream.null.synchronize()
        return outputs


def get_driver():
    return ChainerDriver()
