import chainer
import nnvm
import nnvm.compiler
import onnx
import tvm

from tvm.contrib import graph_runtime

import driver
import utils


class TVMDriver(driver.Driver):
    def name(self):
        return 'tvm'

    def run_first(self, task, inputs, sample_outputs):
        onnx_model = onnx.load_model(task.get_onnx_file())
        symbol, params = nnvm.frontend.from_onnx(onnx_model)
        input_names = symbol.list_input_names()
        assert len(input_names) == len(inputs) + len(params), input_names
        target = 'cuda'

        shape_dict = {}
        dtype_dict = {}
        for name, value in zip(input_names, inputs):
            shape_dict[name] = value.shape
            dtype_dict[name] = value.dtype
        for name, value in params.items():
            shape_dict[name] = value.shape
            dtype_dict[name] = value.dtype
        with nnvm.compiler.build_config(opt_level=3):
            graph, lib, params = nnvm.compiler.build(symbol, target,
                                                     shape=shape_dict,
                                                     dtype=dtype_dict,
                                                     params=params)

        ctx = tvm.gpu()
        # Prepare inputs/outputs.
        tvm_inputs = []
        for input in inputs:
            tvm_inputs.append(tvm.nd.array(input, ctx=ctx))
        tvm_outputs = []
        for output in sample_outputs:
            tvm_outputs.append(tvm.nd.empty(
                output.shape, output.dtype, ctx=ctx))
        graph_module = graph_runtime.create(graph, lib, ctx)

        self.input_names = input_names
        self.params = {k: tvm.nd.array(v, ctx=ctx) for k, v in params.items()}
        self.tvm_inputs = tvm_inputs
        self.tvm_outputs = tvm_outputs
        self.graph_module = graph_module

        tvm_outputs = self.run_task()
        outputs = [o.asnumpy() for o in tvm_outputs]
        return outputs

    def run_task(self):
        inputs = dict(zip(self.input_names, self.tvm_inputs))
        inputs.update(self.params)
        self.graph_module.run(**inputs)
        outputs = []
        for i, output in enumerate(self.tvm_outputs):
            outputs.append(self.graph_module.get_output(i, output))
        return outputs

    def need_onnx(self):
        return True


def get_driver():
    return TVMDriver()
