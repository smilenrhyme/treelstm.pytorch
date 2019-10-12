from __future__ import division
from __future__ import print_function

import os
import math

import torch

from treelstm import Constants
from .vocab import Vocab
import codecs
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import numpy as np

import tempfile, platform
if platform.system() == 'Linux':
    tempfile.tempdir = '/mnt/tmp'


# loading GLOVE word vectors
# if .pth file is found, will load that
# else will load from .txt file & save
def load_word_vectors(path):
    if os.path.isfile(path + '.pth') and os.path.isfile(path + '.vocab'):
        print('==> File found, loading to memory')
        vectors = torch.load(path + '.pth')
        vocab = Vocab(filename=path + '.vocab')
        return vocab, vectors
    # saved file not found, read from txt file
    # and create tensors for word vectors
    print('==> File not found, preparing, be patient')
    special_symbols = [Constants.PAD_WORD, Constants.UNK_WORD,
                       Constants.BOS_WORD, Constants.EOS_WORD]
    count = sum(1 for line in codecs.open(path + '.txt', 'r', encoding='utf8', errors='ignore'))
    count += len(special_symbols)
    with open(path + '.txt', 'r') as f:
        contents = f.readline().rstrip('\n').split(' ')
        dim = len(contents[1:])
    words = [None] * (count)
    vectors = torch.zeros(count, dim, dtype=torch.float, device='cpu')
    with codecs.open(path + '.txt', 'r', encoding='utf8', errors='ignore') as f:
        for idx, item in enumerate(special_symbols):
            vectors[idx].zero_()

        idx = len(special_symbols)
        for line in f:
            contents = line.rstrip('\n').split(' ')
            words[idx] = contents[0]
            values = list(map(float, contents[1:]))
            vectors[idx] = torch.tensor(values, dtype=torch.float, device='cpu')
            idx += 1
    with codecs.open(path + '.vocab', 'w', encoding='utf8', errors='ignore') as f:
        for word in words:
            if word:
                f.write(word + '\n')
    vocab = Vocab(filename=path + '.vocab', data=[Constants.PAD_WORD, Constants.UNK_WORD,
                                                  Constants.BOS_WORD, Constants.EOS_WORD])
    torch.save(vectors, path + '.pth')
    return vocab, vectors


def load_keyed_vectors(path):
    tmp_file = get_tmpfile("_temp_w2v.txt")
    print("tmp_file : ")
    print(tmp_file)
    _ = glove2word2vec(path + '.txt', tmp_file)
    embedding = KeyedVectors.load_word2vec_format(tmp_file)
    print("Embedding Vectors are loaded !!\n")
    # labelToIndex = {w: i for i, w in enumerate(embedding.index2word)}
    # idxToLabel = {i: w for i, w in enumerate(embedding.index2word)}
    dim = embedding.vector_size

    special_symbols = [Constants.PAD_WORD, Constants.UNK_WORD, Constants.BOS_WORD, Constants.EOS_WORD]

    # Build Vocab
    vocab = Vocab()
    vocab.addSpecials(special_symbols)

    for word in embedding.index2word:
        vocab.add(word)

    final_weight_matrix = np.zeros((len(special_symbols), dim))
    final_weight_matrix = np.concatenate((final_weight_matrix, embedding.vectors), axis=0)
    print("Vocab len : ")
    print(len(vocab.labelToIdx))
    print(vocab.idxToLabel[0], vocab.idxToLabel[1], vocab.idxToLabel[2], vocab.idxToLabel[3])
    print("Size of emb matrix : ")
    print(final_weight_matrix.shape)
    print(final_weight_matrix[0:5])
    return vocab, final_weight_matrix

# write unique words from a set of files to a new file
def build_vocab(filenames, vocabfile):
    vocab = set()
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.rstrip('\n').split(' ')
                vocab |= set(tokens)
    with open(vocabfile, 'w') as f:
        for token in sorted(vocab):
            f.write(token + '\n')


# mapping from scalar to vector
def map_label_to_target(label, num_classes):
    target = torch.zeros(1, num_classes, dtype=torch.float, device='cpu')
    ceil = int(math.ceil(label))
    floor = int(math.floor(label))
    if ceil == floor:
        target[0, floor - 1] = 1
    else:
        target[0, floor - 1] = ceil - label
        target[0, ceil - 1] = label - floor
    return target
