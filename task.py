import glob
import importlib
import os

import chainer

import utils


def _makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


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
        self.model_dir = os.path.join('out/models', self.name)
        self.onnx_dir = None

    def run(self):
        _makedirs(self.model_dir)
        param_filename = os.path.join(self.model_dir, 'params.npz')
        params_loaded = self.is_up_to_date(param_filename)
        if params_loaded:
            chainer.serializers.load_npz(param_filename, self.model)

        chainer.config.train = False
        self.model.to_gpu()
        inputs = utils.as_list(self.model.inputs())
        gpu_inputs = utils.to_gpu(inputs)
        gpu_outputs = self.model(*gpu_inputs)
        gpu_outputs = utils.as_list(gpu_outputs)
        outputs = utils.to_cpu(gpu_outputs)
        self.inputs = inputs
        self.outputs = outputs

        if not params_loaded:
            chainer.serializers.save_npz(param_filename, self.model)

        return inputs, outputs

    def is_up_to_date(self, filename):
        if not os.path.exists(filename):
            return False
        return (os.stat(filename).st_mtime >=
                os.stat(self.py_filename).st_mtime)

    def get_onnx_dir(self):
        import onnx
        import onnx_chainer

        if self.onnx_dir is not None:
            return self.onnx_dir
        self.onnx_dir = self.model_dir
        data_dir = os.path.join(self.onnx_dir, 'test_data_set_0')
        _makedirs(data_dir)

        onnx_filename = os.path.join(self.onnx_dir, 'model.onnx')
        if self.is_up_to_date(onnx_filename):
            return self.onnx_dir

        onnx_chainer.export(self.model, list(self.inputs),
                            filename=onnx_filename + '.tmp',
                            graph_name=self.name)

        onnx_model = onnx.load(onnx_filename + '.tmp')
        initializer_names = set(i.name for i in onnx_model.graph.initializer)
        input_names = []
        for input in onnx_model.graph.input:
            if input.name not in initializer_names:
                input_names.append(input.name)
        output_names = []
        for output in onnx_model.graph.output:
            output_names.append(output.name)

        assert len(input_names) == len(self.inputs)
        assert len(output_names) == len(self.outputs)
        for typ, names, values in [('input', input_names, self.inputs),
                                   ('output', output_names, self.outputs)]:
            for i, (name, value) in enumerate(zip(names, values)):
                dtype = onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[value.dtype]
                tensor = onnx.helper.make_tensor(
                    name, dtype, value.shape, value.ravel())
                pb_name = '%s_%d.pb' % (typ, i)
                with open(os.path.join(data_dir, pb_name), 'wb') as f:
                    f.write(tensor.SerializeToString())

        # Commit.
        os.rename(onnx_filename + '.tmp', onnx_filename)
        return self.onnx_dir

    def get_onnx_file(self):
        return os.path.join(self.get_onnx_dir(), 'model.onnx')

    def finish(self):
        """Releases memory."""
        del self.model
        del self.inputs
        del self.outputs


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
