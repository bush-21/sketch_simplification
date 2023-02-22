import torch
import os
from torch.nn import Sequential, Module
from torchvision import transforms
from torchvision.utils import save_image

from PIL import Image, ImageStat
import argparse

parser = argparse.ArgumentParser(description='Sketch simplification demo.')
parser.add_argument('--model', type=str, default='model_gan', help='Model to use.')
# parser.add_argument('--img', type=str, default='test.png', help='Input image file.')
# parser.add_argument('--out', type=str, default='out.png', help='File to output.')
parser.add_argument('--input_path', type=str, required=True, help='Input image path.')
parser.add_argument('--out_path', type=str, required=True, help='Path to output.')
opt = parser.parse_args()
if not(os.path.exists(opt.out_path)):
    os.mkdir(opt.out_path)
model_import = __import__(opt.model, fromlist=['model', 'immean', 'imstd'])
model = model_import.model
immean = model_import.immean
imstd = model_import.imstd

use_cuda = torch.cuda.device_count() > 0

model.load_state_dict(torch.load(opt.model + ".pth"))
model.eval()

#for filename in os.listdir(r"./"+opt.input_path):
for filename in os.listdir(opt.input_path):
    data = Image.open(opt.input_path+'/'+filename).convert('L')
    w, h = data.size[0], data.size[1]
    pw = 8 - (w % 8) if w % 8 != 0 else 0
    ph = 8 - (h % 8) if h % 8 != 0 else 0
    stat = ImageStat.Stat(data)

    data = ((transforms.ToTensor()(data) - immean) / imstd).unsqueeze(0)
    if pw != 0 or ph != 0:
        data = torch.nn.ReplicationPad2d((0, pw, 0, ph))(data).data

    if use_cuda:
        pred = model.cuda().forward(data.cuda()).float()
    else:
        pred = model.forward(data)
    save_image(pred[0], opt.out_path+'/'+filename)
