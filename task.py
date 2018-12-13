import glob
import importlib
import os

import chainer


class Task(object):
    def __init__(self, model):
        """Initializes the task object.

        Args:
          model: A `chainer.Chain` with additional required attributes:
            name: A str of the task name.
            inputs: A np.array or a tuple of np.array objects to be
              fed to `forward` function.
        """
        assert isinstance(model, chainer.Chain), model
        assert hasattr(model, 'inputs')
        self.name = model.name
        self.model = model

    def run(self):
        self.model.to_gpu()
        inputs = self.model.inputs()
        if not isinstance(inputs, tuple):
            inputs = [inputs]
        gpu_inputs = [self.model.xp.array(input) for input in inputs]
        gpu_outputs = self.model(*gpu_inputs)
        if not isinstance(gpu_outputs, tuple):
            gpu_outputs = [gpu_outputs]
        outputs = [chainer.cuda.to_cpu(v.array) for v in gpu_outputs]
        return inputs, outputs


def import_file(filename):
    module_name = filename[:-3].replace('/', '.')
    return importlib.import_module(module_name)


def collect_all_tasks():
    task_pys = sorted(glob.glob('tasks/*.py'))
    task_pys += sorted(glob.glob('*/tasks/*.py'))
    tasks = []
    for task_py in task_pys:
        module = import_file(task_py)
        for task in module.get_tasks():
            tasks.append(Task(task))
    return tasks
