import tensorflow as tf
import onnx
import onnx_tf

import driver


class TFDriver(driver.Driver):
    def name(self):
        return 'tf_cpu'

    def run_first(self, task, inputs, sample_outputs):
        onnx_filename = task.get_onnx_file()
        onnx_model = onnx.load(onnx_filename)
        self.session = onnx_tf.backend.prepare(onnx_model)
        self.inputs = inputs
        outputs = self.run_task(task)
        return outputs

    def run_task(self, task):
        outputs = self.session.run(*self.inputs)
        return outputs

    def need_onnx(self):
        return True


def get_driver():
    return TFDriver()
