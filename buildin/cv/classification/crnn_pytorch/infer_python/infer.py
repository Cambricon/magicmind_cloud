from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import dataset
import util
import magicmind.python.runtime as mm
from PIL import Image
import linecache
import sys
sys.path.append("../../../utils")
from utils import Record
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir + "/../export_model/crnn.pytorch")
import models.crnn as crnn

def get_line_context(file_path, line_number):
    return linecache.getline(file_path, line_number).strip()
parser = argparse.ArgumentParser()
parser = argparse.ArgumentParser()
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--file_path', required=True, help='path to lexicon.txt')
parser.add_argument('--magicmind_model', required=True, help='path to mm model')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--infr_iter', type=int, default=1000, help='number of iters to infr for')
parser.add_argument('--top1_file', type=str,default="../data/output/result.txt", help='path to save result')
opt = parser.parse_args()
print(opt)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

test_dataset = dataset.lmdbDataset(
    root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))

print("WARNING: warpctc is not supported !")
from torch.nn import CTCLoss
cudnn.enabled = False
criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)

image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)

image = Variable(image)
text = Variable(text)
length = Variable(length)

print('Start infer')
model_path = opt.magicmind_model
model = mm.Model()
model.deserialize_from_file(model_path)

dev = mm.Device()
dev.id = 0
dev.active()

converter = util.strLabelConverter(opt.alphabet)
data_loader = torch.utils.data.DataLoader(
    test_dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
val_iter = iter(data_loader)
epoch_size = len(data_loader)

n_correct = 0
loss_avg = util.averager()
print("epoch_size:",epoch_size)

max_iter = min(opt.infr_iter, epoch_size)
for i in range(max_iter):
    data = val_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    util.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    util.loadData(text, t)
    util.loadData(length, l)

    econfig = mm.Model.EngineConfig()
    econfig.device_type = "MLU"
    engine = model.create_i_engine(econfig)
    assert engine != None, "Failed to create engine"
    queue = dev.create_queue()
    assert queue != None
    context = engine.create_i_context()

    ###input
    inputs = context.create_inputs()
    outputs = []
    inputs[0].from_numpy(image.numpy())
    ###enqueue
    status = context.enqueue(inputs, outputs, queue)
    assert status.ok(), status.error_message()
    queue.sync()
    ###output
    outputs[0] = torch.tensor(outputs[0].asnumpy().astype(np.float32))
    preds = outputs[0]
    ###postprecess
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
    if batch_size == 1:
        sim_preds=[sim_preds]

    for pred, target in zip(sim_preds, cpu_texts):
        target=get_line_context(opt.file_path, int(target)+1)
        if pred == target.lower():
            n_correct += 1

raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
if batch_size == 1:
    raw_preds=[raw_preds]
for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
    print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, get_line_context(opt.file_path, int(gt)+1)))

accuracy = n_correct / float(max_iter * opt.batchSize)
result_top1 = Record(opt.top1_file)
result_top1.write("top1 accuracy: %f"%(accuracy), True)
