import glob
import importlib
import os

import chainer
import onnx_chainer


class Task(object):
    def __init__(self, model):
        """Initializes the task object.

        Args:
          model: A `chainer.Chain` with additional required attributes:
            category: A str of the category of the task.
            name: A str of the task name.
            inputs: A np.array or a tuple of np.array objects to be
              fed to `forward` function.
        """
        assert isinstance(model, chainer.Chain), model
        assert hasattr(model, 'inputs')
        self.name = model.name
        self.model = model
        self.onnx_dir = None

    def run(self):
        chainer.config.train = False
        self.model.to_gpu()
        inputs = self.model.inputs()
        if not isinstance(inputs, tuple):
            inputs = [inputs]
        gpu_inputs = [self.model.xp.array(input) for input in inputs]
        gpu_outputs = self.model(*gpu_inputs)
        if not isinstance(gpu_outputs, tuple):
            gpu_outputs = [gpu_outputs]
        outputs = [chainer.cuda.to_cpu(v.array) for v in gpu_outputs]
        self.inputs = inputs
        return inputs, outputs

    def get_onnx_dir(self):
        if self.onnx_dir is not None:
            return self.onnx_dir
        self.onnx_dir = os.path.join('out/onnx', self.name)
        if not os.path.exists(self.onnx_dir):
            os.makedirs(self.onnx_dir)

        # TODO(hamaji): Cache ONNX model files.

        onnx_chainer.export(self.model, self.inputs,
                            filename='%s/model.onnx' % self.onnx_dir,
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
            tasks.append(Task(task))
    return tasks
