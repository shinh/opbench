import chainer

import driver


class ChainerDriver(driver.Driver):
    def prepare_task(self, task):
        task.model.to_gpu()
        inputs = self.model.inputs()
        if not isinstance(inputs, tuple):
            inputs = [inputs]
        self.inputs = [self.model.xp.array(input) for input in inputs]

    def run_task(self, task):
        return task.model(*inputs)

    def get_result(self, task):
        return chainer.cuda.to_cpu(self.run_task(task))
