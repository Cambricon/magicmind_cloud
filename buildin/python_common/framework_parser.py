from magicmind.python.runtime import DataType, ModelKind, Network
from magicmind.python.runtime.parser import Parser

from logger import Logger

log = Logger()


class TensorFlowParser(object):
    def __init__(self, args):
        self.parser = Parser(ModelKind.kTensorflow)
        self.args = args

    def parse_model(self, network: Network) -> Network:
        """Parse model.

        Args:
            network(Network): Out parameter, the parsed network.

        Returns:
            - Returns a parsed network.
        """
        self.parser.set_model_param("tf-model-type", "tf-graphdef-file")
        self.parser.set_model_param("tf-graphdef-inputs", self.args.input_names)
        self.parser.set_model_param("tf-graphdef-outputs", self.args.output_names)
        self.parser.set_model_param("tf-infer-shape", True)
        self.parser.parse(network, self.args.tf_pb)
        return network

    def parse(self) -> Network:
        """Parse framework model."""
        network = Network()
        self.parse_model(network)
        log.info("TensorFlowModelParser: input model was parsed successfully.")
        return network


class PytorchParser(object):
    def __init__(self, args):
        self.parser = Parser(ModelKind.kPytorch)
        self.args = args

    def parse_model(self, network: Network) -> Network:
        """Parse model.

        Args:
            network(Network): Out parameter, the parsed network.

        Returns:
            - Returns a parsed network.
        """
        DATATYPE_TO_MM = {
            "INT8": DataType.INT8,
            "INT16": DataType.INT16,
            "INT32": DataType.INT32,
            "UINT8": DataType.UINT8,
            "UINT16": DataType.UINT16,
            "UINT32": DataType.UINT32,
            "HALF": DataType.FLOAT16,
            "FLOAT": DataType.FLOAT32,
            "BOOL": DataType.BOOL,
            "QINT8": DataType.QINT8,
            "QINT16": DataType.QINT16,
        }

        # there may have multiple inputs
        # dt = [self.args["pt_input_dtypes"]] if isinstance(self.args["pt_input_dtypes"],
        #                                                  str) else self.args["pt_input_dtypes"]
        dt = (
            [self.args.pt_input_dtypes]
            if isinstance(self.args.pt_input_dtypes, str)
            else self.args.pt_input_dtypes
        )
        self.parser.set_model_param(
            "pytorch-input-dtypes", [DATATYPE_TO_MM[t] for t in dt]
        )
        self.parser.parse(network, self.args.pytorch_pt)
        # self.parser.parse(network, self.args["pytorch_pt"])
        return network

    def parse(self) -> Network:
        """Parse framework model."""
        network = Network()
        self.parse_model(network)
        log.info("PytorchModelParser: input model was parsed successfully.")
        return network


class CaffeParser(object):
    def __init__(self, args):
        self.parser = Parser(ModelKind.kCaffe)
        self.args = args

    def parse_model(self, network: Network) -> Network:
        """Parse model.

        Args:
            network(Network): Out parameter, the parsed network.

        Returns:
            - Returns a parsed network.
        """
        self.parser.parse(network, self.args.caffemodel, self.args.prototxt)
        return network

    def parse(self) -> Network:
        """Parse framework model."""
        network = Network()
        self.parse_model(network)
        log.info("CaffeModelParser: input model was parsed successfully.")
        return network


class OnnxParser(object):
    def __init__(self, args):
        self.parser = Parser(ModelKind.kOnnx)
        self.args = args

    def parse_model(self, network: Network) -> Network:
        """Parse model.

        Args:
            network(Network): Out parameter, the parsed network.

        Returns:
            - Returns a parsed network.
        """
        self.parser.parse(network, self.args.onnx)
        return network

    def parse(self) -> Network:
        """Parse framework model."""
        network = Network()
        self.parse_model(network)
        log.info("OnnxModelParser: input model was parsed successfully.")
        return network
