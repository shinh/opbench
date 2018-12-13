import glob
import importlib
import os

import chainer
import onnx_chainer

import utils


class Task(object):
    def __init__(self, model, py_filename):
        """Initializes the task object.

        Args:
          model: A `chainer.Chain` with additional required attributes:
            category: A str of the category of the task.
            name: A str of the task name.
            inputs: A np.array or a tuple of np.array objects to be
              fed to `forward` function.
          py_filename: A str object.
        """
        assert isinstance(model, chainer.Chain), model
        assert hasattr(model, 'inputs')
        self.name = model.name
        self.model = model
        self.py_filename = py_filename
        self.onnx_dir = None

    def run(self):
        chainer.config.train = False
        self.model.to_gpu()
        inputs = utils.as_list(self.model.inputs())
        gpu_inputs = utils.to_gpu(inputs)
        gpu_outputs = self.model(*gpu_inputs)
        gpu_outputs = utils.as_list(gpu_outputs)
        outputs = utils.to_cpu(gpu_outputs)
        self.inputs = inputs
        return inputs, outputs

    def get_onnx_dir(self):
        if self.onnx_dir is not None:
            return self.onnx_dir
        self.onnx_dir = os.path.join('out/onnx', self.name)
        if not os.path.exists(self.onnx_dir):
            os.makedirs(self.onnx_dir)

        onnx_filename = os.path.join(self.onnx_dir, 'model.onnx')
        if (os.path.exists(onnx_filename) and
            (os.stat(onnx_filename).st_mtime >=
             os.stat(self.py_filename).st_mtime)):
            return self.onnx_dir

        onnx_chainer.export(self.model, self.inputs,
                            filename=onnx_filename,
                            graph_name=self.name)
        return self.onnx_dir

    def get_onnx_file(self):
        return os.path.join(self.get_onnx_dir(), 'model.onnx')


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
            tasks.append(Task(task, task_py))
    return tasks
