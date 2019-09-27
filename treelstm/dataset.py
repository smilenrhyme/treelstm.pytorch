import os
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.utils.data as data

from . import Constants
from .tree import Tree, ConstTree


# Dataset class for SICK dataset
class SICKDataset(data.Dataset):
    def __init__(self, path, vocab, num_classes, use_parse_tree):
        super(SICKDataset, self).__init__()
        self.vocab = vocab
        self.num_classes = num_classes

        self.lsentences = self.read_sentences(os.path.join(path, 'a.toks'))
        self.rsentences = self.read_sentences(os.path.join(path, 'b.toks'))

        if use_parse_tree == 'dependency':
            self.ltrees = self.read_dep_trees(os.path.join(path, 'a.parents'))
            self.rtrees = self.read_dep_trees(os.path.join(path, 'b.parents'))
        elif use_parse_tree == 'constituency':
            self.ltrees = self.read_constituency_trees(os.path.join(path, 'a.cparents'), os.path.join(path, 'a.toks'))
            self.rtrees = self.read_constituency_trees(os.path.join(path, 'b.cparents'), os.path.join(path, 'b.toks'))

        self.labels = self.read_labels(os.path.join(path, 'sim.txt'))

        self.size = self.labels.size(0)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        ltree = deepcopy(self.ltrees[index])
        rtree = deepcopy(self.rtrees[index])
        lsent = deepcopy(self.lsentences[index])
        rsent = deepcopy(self.rsentences[index])
        label = deepcopy(self.labels[index])
        return (ltree, lsent, rtree, rsent, label)

    def read_sentences(self, filename):
        with open(filename, 'r') as f:
            sentences = [self.read_sentence(line) for line in tqdm(f.readlines())]
        return sentences

    def read_sentence(self, line):
        indices = self.vocab.convertToIdx(line.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def read_dep_trees(self, filename):
        with open(filename, 'r') as f:
            trees = [self.read_dep_tree(line) for line in tqdm(f.readlines())]
        return trees

    @staticmethod
    def read_dep_tree(line):
        parents = list(map(int, line.split()))
        trees = dict()
        root = None
        for i in range(1, len(parents) + 1):
            if i - 1 not in trees.keys() and parents[i - 1] != -1:
                idx = i
                prev = None
                while True:
                    parent = parents[idx - 1]
                    if parent == -1:
                        break
                    tree = Tree()
                    if prev is not None:
                        tree.add_child(prev)
                    trees[idx - 1] = tree
                    tree.idx = idx - 1
                    if parent - 1 in trees.keys():
                        trees[parent - 1].add_child(tree)
                        break
                    elif parent == 0:
                        root = tree
                        break
                    else:
                        prev = tree
                        idx = parent
        return root

    def read_constituency_trees(self, parents_path, tokens_path):
        const_trees, parents, toks = [], [], []

        with open(tokens_path, 'r') as tp:
            for line in tp:
                toks.append(line.strip().split())

        parents = []
        with open(parents_path, 'r') as pp:
            for line in pp:
                parents.append(map(int, line.split()))

        const_trees = [self.read_constituency_tree(parents[i], toks[i]) for i in tqdm(xrange(len(toks)))]
        return const_trees

    @staticmethod
    def read_constituency_tree(parents, words):
        trees = []
        root = None
        size = len(parents)
        for i in xrange(size):
            trees.append(None)

        # word_idx = 0
        for i in xrange(size):
            if not trees[i]:
                idx = i
                prev = None
                prev_idx = None
                # word = words[word_idx]
                # word_idx += 1
                while True:
                    tree = ConstTree()
                    parent = parents[idx] - 1
                    # tree.word = word
                    tree.parent, tree.idx = parent, idx
                    word = None
                    if prev is not None:
                        if tree.left is None:
                            tree.left = prev
                        else:
                            tree.right = prev
                    trees[idx] = tree
                    if parent >= 0 and trees[parent] is not None:
                        if trees[parent].left is None:
                            trees[parent].left = tree
                        else:
                            trees[parent].right = tree
                        break
                    elif parent == -1:
                        root = tree
                        break
                    else:
                        prev = tree
                        prev_idx = idx
                        idx = parent

        leaf_idx = 0
        for i in xrange(size):
            tree = trees[i]
            if tree is not None and tree.left is None and tree.right is None:
                tree.leaf_idx = leaf_idx
                leaf_idx = leaf_idx + 1

        return root

    @staticmethod
    def read_labels(filename):
        with open(filename, 'r') as f:
            labels = list(map(lambda x: float(x), f.readlines()))
            labels = torch.tensor(labels, dtype=torch.float, device='cpu')
        return labels
