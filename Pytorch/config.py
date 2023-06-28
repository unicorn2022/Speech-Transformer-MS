import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
input_dim = 80  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT (ms)
hidden_size = 512
embedding_dim = 512
cmvn = True  # apply CMVN on feature
num_layers = 4
LFR_m = 4
LFR_n = 3
sample_rate = 16000  # aishell

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 100  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
sos_id = 0
eos_id = 1
num_train = 28539
num_dev = 2703
num_test = 2620
vocab_size = 35663

DATA_DIR = 'data'
librispeech_folder = 'data/LibriSpeech'
wav_folder = os.path.join(librispeech_folder, 'wav')
tran_file = os.path.join(librispeech_folder, 'transcript/transcript.txt')
pickle_file = 'data/librispeech.pickle'
