import numpy as np
import torch

from mm_runner import MMRunner
from logger import Logger

log = Logger()
class MM_Model(object):
    def __init__(self, model_path):
        #mlu
        model = mm.Model()
        model.deserialize_from_file(model_path)
        
        dev = mm.Device()
        dev.id = 0
        dev.active()
        
        with mm.System():
            econfig = mm.Model.EngineConfig()
            econfig.device_type = "MLU"
            engine = model.create_i_engine(econfig)
            assert engine != None, "Failed to create engine"
            self.queue = dev.create_queue()
            assert self.queue != None
            self.context = engine.create_i_context()
            self.inputs = self.context.create_inputs()

    def forward_db(self, input):
        outputs = []
        host_out = []
        self.inputs[0].from_numpy(input.numpy())
        self.context.enqueue(self.inputs, outputs, self.queue)
        self.queue.sync()
        host_out = torch.tensor(outputs[0].asnumpy())
        return host_out

def main(args):
    dbnet = MMRunner(mm_file=args.magicmind_model, device_id=args.device_id)    
    input = np.random.randint(255,size=(args.batch_size, 3, args.input_height, args.input_width)).astype(np.float32)
    
    outputs = dbnet(input)
    db_output = outputs[0].astype(np.float32)

if __name__ == '__main__':
    parser.add_argument('--magicmind_model', type=str, default = '../../data/models/dbnet_pt_model_force_float32_1_1280_800_model', help='model path')
    parser.add_argument("--device_id", "--device_id", type=int, default=0, help="device_id")
    parser.add_argument('--batch_size', dest = 'batch_size', default = 1, type = int, help = 'input batch size')
    parser.add_argument('--input_width', dest = 'input_width', default = 800, type = int, help = 'model input width')
    parser.add_argument('--input_height', dest = 'input_height', default = 1280, type = int, help = 'model input height')
    args = parser.parse_args()
    if not os.path.exists(args.magicmind_model):
        log.error(args.magicmind_model + " does not exist.")
        exit()
    main(args)
