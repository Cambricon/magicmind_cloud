from insightface.recognition.arcface_torch.backbones import get_model
import argparse
import torch

def main():
    args = argparse.ArgumentParser(description='save arcface model by jit.trace')
    args.add_argument('--weights', default = './../data/models/backbone.pth',
            required=True, type = str, help = 'arcface weights')
    args.add_argument('--network', default = 'r100',
            type = str, help = 'backbone network name. see https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch for supported networks')
    args.add_argument('--input_width', dest = 'input_width', default = 112,
            type = int, help = 'model input width')
    args.add_argument('--input_height', dest = 'input_height', default = 112,
            type = int, help = 'model input height')
    args.add_argument('--output_pt', default = './../data/models/arcface.pt',
            type = str, help = 'output pt path')

    args = args.parse_args()

    input_data = torch.randn(1, 3, args.input_height, args.input_width).float()
    net = get_model(args.network, fp16=False)
    net.load_state_dict(torch.load(args.weights, map_location='cpu'))
    net.eval()
    traced = torch.jit.trace(net, input_data)
    torch.jit.save(traced, args.output_pt)
    print('torchscript saved to', args.output_pt)

if __name__ == "__main__":
    main()

