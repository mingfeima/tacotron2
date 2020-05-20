#!/usr/bin/env python
# encoding: utf-8

import argparse
import sys
import numpy as np
import torch
from time import time

from hparams import create_hparams
from model import Tacotron2
# from layers import TacotronSTFT, STFT
from train import load_model
from text import text_to_sequence

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Tacotron2 inference')
parser.add_argument('--cuda', action='store_true', help='with CUDA')
parser.add_argument('--profile', action='store_true', help='Profile')
parser.add_argument('--quantize', action='store_true', help='Quantize')
args = parser.parse_args()
cuda = True if torch.cuda.is_available() and args.cuda else False
device = torch.device('cuda' if cuda else 'cpu')
if not torch.cuda.is_available():
    print('CUDA is not available.')
print('Device: {}'.format(device))

# Setup hparams
hparams = create_hparams()
hparams.sampling_rate = 22050

# Load model from checkpoint
checkpoint_path = "models/tacotron2_statedict.pt"
model = load_model(hparams, device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device)['state_dict'])
# _ = model.cuda().eval().half()
model.to(device)
model.eval()

if args.quantize:
    # quantization model
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8)

print(model)

# Prepare text input
text = "Waveglow is really awesome!"
sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).long().to(device)

# Decode text input
niters = 100
t0 = time()
with torch.autograd.profiler.profile(enabled=args.profile, use_cuda=cuda) as prof:
    with torch.no_grad():
        for i in range(niters):
            mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
t1 = time()
ttime = (t1 - t0) /niters * 1000

if args.profile:
    print(prof.key_averages().table(sort_by='self_cpu_time_total'))

print('Ave exec time: {:.2f} ms'.format(ttime))
