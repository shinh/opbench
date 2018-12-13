import time

import chainer
import numpy as np


class Driver(object):
    def __init__(self):
        pass

    def bench(self, task, time_budget_sec=1.0, max_count=100):
        inputs, expected_outputs = task.run()
        actual_outputs = self.get_result(task, inputs)
        np.testing.assert_allclose(expected_outputs, actual_outputs, rtol=1e-2)

        times = []
        start_time = time.time()
        for t in range(max_count):
            now = time.time()
            times.append(now)
            if now - start_time >= time_budget_sec:
                break
            self.run_task(task)

        result = []
        for i in range(len(times) - 1):
            result.append(times[i + 1] - times[i])
        return result

    def prepare_task(self, task):
        pass

    def get_result(self, task):
        raise NotImplementedError(
            '`get_result` must be overridden for %s' % type(self))

    def run_task(self, task):
        raise NotImplementedError(
            '`run_task` must be overridden for %s' % type(self))
