import os
import magicmind.python.runtime as mm

from logger import Logger

log = Logger()


class MMRunner:
    """Wrapper for detector's inference with MagicMindRuntime."""

    def __init__(self, mm_file, device_id):
        super(MMRunner, self).__init__()
        # check mm_file
        assert os.path.exists(mm_file), "mm model file not exist,please check it."
        assert device_id >= 0, "invalid device_id,please check it."

        self.model = mm.Model()
        self.model.deserialize_from_file(mm_file)
        log.info("Model instance Created Success!")

        with mm.System() as mm_sys:
            dev_count = mm_sys.device_count()
            assert dev_count > 0, "dev_count must be large than 0"

            # check device_id
            assert device_id < dev_count, "deviced_id must be less than dev_count"

            self.dev = mm.Device()
            self.dev.id = device_id
            assert self.dev.active().ok()
            log.info("Model dev Created Success!")

            self.econfig = mm.Model.EngineConfig()
            self.econfig.device_type = "MLU"

            self.engine = self.model.create_i_engine(self.econfig)
            assert self.engine != None, "Failed to create engine"
            log.info("Model engine Created Success!")

            self.context = self.engine.create_i_context()
            assert self.context != None, "Failed to create context"
            log.info("Model context Created Success!")

            self.queue = self.dev.create_queue()
            assert self.queue != None, "Failed to create queue"
            log.info("Model queue Created Success!")

            self.inputs = self.context.create_inputs()
            log.info("Model inputs Created Success!")
            log.info("All Model resource Created Success!")

            self.input_nums = self.model.get_input_num()
            self.outputs = []

    def __del__(self):
        _all_attrs = dir(self)

        if "inputs" in _all_attrs:
            for t in self.inputs:
                del t
            log.info("Model inputs Destoryed Success!")

        if "context" in _all_attrs:
            del self.context
            log.info("Model context Destoryed Success!")

        if "engine" in _all_attrs:
            del self.engine

        if "model" in _all_attrs:
            del self.model
            log.info("Model instance Destoryed Success!")

        if "queue" in _all_attrs:
            del self.queue
            log.info("Model queue Destoryed Success!")

        if "dev" in _all_attrs:
            del self.dev
            log.info("Model dev Destoryed Success!")

        log.info("All Model resource Destoryed Success!")

    def __call__(self, inputs, **kwargs):
        # check inputs
        assert isinstance(
            inputs, list
        ), "inputs data type must be list,please check it."

        # model infer
        for _input_idx in range(self.input_nums):
            self.inputs[_input_idx].from_numpy(inputs[_input_idx])
            self.inputs[_input_idx].to(self.dev)

        self.outputs = []

        assert self.context.enqueue(self.inputs, self.outputs, self.queue).ok()
        assert self.queue.sync().ok()

        # get output
        outputs = []
        for _output_idx in range(self.model.get_output_num()):
            outputs.append(self.outputs[_output_idx].asnumpy())

        return outputs
