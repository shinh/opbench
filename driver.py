import time

import chainer
import numpy as np


class Driver(object):
    def bench(self, task, time_budget_sec=1.0, max_count=100):
        inputs, expected_outputs = task.run()
        actual_outputs = self.run_first(task, inputs)
        np.testing.assert_allclose(expected_outputs, actual_outputs, rtol=1e-2)

        times = []
        start_time = time.time()
        for t in range(max_count):
            now = time.time()
            times.append(now)
            if now - start_time >= time_budget_sec:
                break
            self.run_task(task)
        times.append(time.time())

        result = []
        for i in range(len(times) - 1):
            result.append(times[i + 1] - times[i])
        return result

    def run_first(self, task, inputs):
        """Runs the model first time.

        This run is both for testing sanity of the driver and to give
        a chance for drivers to warm-up the model and stage inputs to
        the device.

        Args:
          task: A `Task` object.
          inputs: A list of np.array objects.

        Returns:
          A list of np.array objects.
        """
        raise NotImplementedError(
            '`get_result` must be overridden for %s' % type(self))

    def run_task(self, task):
        """Runs the task once.

        The inputs passed to `run_first` in previous run should be
        used as the input of the model.

        Args:
          task: A `Task` object.
        """
        raise NotImplementedError(
            '`run_task` must be overridden for %s' % type(self))
