import chainer
import cupy
import tensorrt

import driver
import utils


class TensorRTDriver(driver.Driver):
    def run_first(self, task, inputs, sample_outputs):
        self.batch_size = inputs[0].shape[0]
        onnx_filename = task.get_onnx_file()
        with open(onnx_filename, 'rb') as f:
            onnx_proto = f.read()

        logger = tensorrt.Logger()
        # logger = tensorrt.Logger(tensorrt.Logger.Severity.INFO)
        builder = tensorrt.Builder(logger)
        builder.max_batch_size = self.batch_size
        network = builder.create_network()
        parser = tensorrt.OnnxParser(network, logger)
        parser.parse(onnx_proto)
        engine = builder.build_cuda_engine(network)
        self.context = engine.create_execution_context()

        assert len(inputs) + len(sample_outputs) == engine.num_bindings
        for i, input in enumerate(inputs):
            assert self.batch_size == input.shape[0]
            assert input.shape[1:] == engine.get_binding_shape(i)
        for i, output in enumerate(sample_outputs):
            assert self.batch_size == output.shape[0]
            i += len(inputs)
            assert output.shape[1:] == engine.get_binding_shape(i)

        self.inputs = utils.to_gpu(inputs)
        self.outputs = []
        for output in sample_outputs:
            self.outputs.append(cupy.zeros_like(output))
        self.run_task(task)
        return utils.to_cpu(self.outputs)

    def run_task(self, task):
        bindings = [a.data.ptr for a in self.inputs]
        bindings += [a.data.ptr for a in self.outputs]
        result = self.context.execute(self.batch_size, bindings)
        assert result
        chainer.cuda.Stream.null.synchronize()


def get_driver():
    return TensorRTDriver()
