import torch
from model import LeNet
from gcommand_loader import spect_loader
import numpy as np
import warnings
import argparse
import os

dirs = os.listdir("./gcommands/train/")
dirs = sorted(dirs)

parser = argparse.ArgumentParser(
    description='ConvNets for Speech Commands Recognition')

parser.add_argument('--wav_path', default='gcommands/recordings/one/3.wav',
                    help='path to the audio file')

args = parser.parse_args()

path = args.wav_path

warnings.filterwarnings("ignore")

model = LeNet()
model.load_state_dict(torch.load("checkpoint/ckpt.t7"))

model.eval()

wav = spect_loader(path, window_size=.02, window_stride=.01, normalize=True, max_len=101, window='hamming')
#print(wav.shape)
with torch.no_grad():
	label = model.forward(wav.view(1,1,161,101))
print(dirs[np.argmax(np.ravel(label.numpy()))])