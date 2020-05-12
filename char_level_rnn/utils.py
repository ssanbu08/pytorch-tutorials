from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch


def find_files(path):
    return glob.glob(path)


print(find_files('/home/anbarasan/Workspace/Anbu/code/pytorch/char_level_rnn/data/names/*.txt'))

all_letters = string.ascii_letters + ".,;'"
n_letters = len(all_letters)


# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
print(unicode_to_ascii('Ślusàrski'))

def letterToIndex(letter):
    return all_letters.find(letter)

# Just for demonstration, turn a letter into a < 1 x n_letters > Tensor
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot vectors
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor