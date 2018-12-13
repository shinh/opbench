import chainer
import cupy
import tensorrt

import driver
import utils


class TensorRTDriver(driver.Driver):
    def run_first(self, task, inputs, sample_outputs):
        onnx_filename = task.get_onnx_file()
        with open(onnx_filename, 'rb') as f:
            onnx_proto = f.read()

        logger = tensorrt.Logger()
        # logger = tensorrt.Logger(tensorrt.Logger.Severity.INFO)
        builder = tensorrt.Builder(logger)
        network = builder.create_network()
        parser = tensorrt.OnnxParser(network, logger)
        parser.parse(onnx_proto)
        engine = builder.build_cuda_engine(network)
        self.context = engine.create_execution_context()

        # for i in range(3):
        #     print(engine.get_binding_name(i))
        #     print(engine.get_binding_dtype(i))
        #     print(engine.get_binding_shape(i))

        self.inputs = utils.to_gpu(inputs)
        self.outputs = []
        for output in sample_outputs:
            self.outputs.append(cupy.zeros_like(output))
        self.run_task(task)
        return utils.to_cpu(self.outputs)

    def run_task(self, task):
        bindings = [a.data.ptr for a in self.inputs]
        bindings += [a.data.ptr for a in self.outputs]
        self.context.execute(self.inputs[0].shape[0], bindings)
        chainer.cuda.Stream.null.synchronize()


def get_driver():
    return TensorRTDriver()
