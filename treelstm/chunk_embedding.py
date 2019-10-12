import torch

from treelstm import SimilarityTreeLSTM, utils
from treelstm import Constants
import random
import numpy as np
import os

from treelstm.tree import ConstTree


class ChunkEmbedding(object):
    def __init__(self, model, pickle_file, vocab, classpath, temp_dir, seed=123):
        self._set_seed(seed)
        self.sim_model = model
        self._load_model(pickle_file)
        self.vocab = vocab
        self.classpath = classpath
        self.temp_dir = temp_dir

    @staticmethod
    def _set_seed(seed):
        # Todo : Ideally seed should come from model pickle file itself
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

    def _load_model(self, pickle_file):
        device = torch.device("cpu")
        self.sim_model.load_state_dict(
            torch.load(pickle_file)['model'])
        self.sim_model.to(device)
        self.sim_model.eval()

    @staticmethod
    def __read_constituency_tree(parents, words):
        trees = []
        root = None
        size = len(parents)
        for i in xrange(size):
            trees.append(None)

        word_idx = 0
        for i in xrange(size):
            if not trees[i]:
                idx = i
                prev = None
                prev_idx = None
                word = words[word_idx]
                word_idx += 1
                while True:
                    tree = ConstTree()
                    parent = parents[idx] - 1
                    tree.word, tree.parent, tree.idx = word, parent, idx
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

    def __convert_to_vocab(self, sentence):
        indices = self.vocab.convertToIdx(sentence.split(), Constants.UNK_WORD)
        return torch.tensor(indices, dtype=torch.long, device='cpu')

    def get_chunk_representation(self, sentence, chunk, const_tree=None, idx_span_mapping=None, mode='non_leaf'):
        if mode == 'leaf':
            with torch.no_grad():
                emb_input = self.sim_model.emb(self.__convert_to_vocab(chunk))
                c, h = self.sim_model.model.leaf_module(torch.squeeze(emb_input))
                return h
        else:
            chunk_string = ' '.join(chunk)
            chunk_idx = None
            for k, v in idx_span_mapping.items():
                if v == chunk_string:
                    chunk_idx = k
                    break

            if chunk_idx is None:
                return "Error in finding chunk index"

            with torch.no_grad():
                emb_input = self.sim_model.emb(self.__convert_to_vocab(sentence))
                idx_state = {}
                self.sim_model.model(const_tree, emb_input, idx_state)
                return idx_state[chunk_idx][1]

    def inOrder(self, root, idx_span_mapping):
        if root:
            self.inOrder(root.left, idx_span_mapping)
            idx_span_mapping[root.idx] = root.span.lower()
            self.inOrder(root.right, idx_span_mapping)

    def constituent_parse(self, sentence, tokenize=False):
        dirpath = os.path.dirname(self.temp_dir)
        parentpath = os.path.join(dirpath, 'temp.cparents')
        tokpath = os.path.join(dirpath, 'temp.toks')
        tokenize_flag = '-tokenize - ' if tokenize else ''
        cmd = ('java -cp %s MinimalConstituencyParse -tokpath %s -parentpath %s %s -sentence %s' % (self.classpath, tokpath, parentpath, tokenize_flag, '"' + sentence + '"'))
        exit_status = os.system(cmd)

        if exit_status != 0:
            raise Exception("Constituent parse failed..")

        with open(tokpath, 'r') as tp:
            for line in tp:
                toks = line.strip().split()

        with open(parentpath, 'r') as pp:
            for line in pp:
                parents = map(int, line.split())

        try:
            const_tree = self.__read_constituency_tree(parents, toks)
            const_tree.set_spans()
            idx_span_mapping = {}
            self.inOrder(const_tree, idx_span_mapping)
            return const_tree, idx_span_mapping
        except Exception as e:
            print "__read_constituency_tree failed :", sentence
            raise Exception("__read_constituency_tree failed")

    def get_embedding(self, sentence, chunk, const_tree, idx_span_mapping):
        sentence = sentence.strip().lower()
        chunk = [e.strip().lower() for e in chunk]

        if len(chunk) == 0 or set(sentence.split()).intersection(set(chunk)) != set(chunk):
            # Invalid chunk for given sentence !!
            return torch.zeros(1, 1)

        if len(chunk) == 1:
            return self.get_chunk_representation(sentence, chunk[0], const_tree, idx_span_mapping, mode='leaf')
        else:
            # check chunk is constituent or not
            if ' '.join(chunk) in idx_span_mapping.values():
                # Get non_leaf tree-LSTM representation
                return self.get_chunk_representation(sentence, chunk, const_tree, idx_span_mapping, mode='non_leaf')
            else:
                # Chunk is not a constituent, returning mean embedding
                h_list = []
                for token in chunk:
                    h_list.append(self.get_chunk_representation(sentence, token, const_tree, idx_span_mapping, mode='leaf'))
                final_h = torch.stack(h_list, dim=0)
                return torch.mean(final_h, dim=0)


if __name__ == '__main__':
    pickle_file = "/Users/manish.bansal/repos/search/treelstm.pytorch/checkpoints/best_model_with_states.pt"

    glove_vocab, glove_emb_matrix = utils.load_keyed_vectors("/Users/manish.bansal/repos/search/treelstm.pytorch/data/glove/sample_glove")

    model = SimilarityTreeLSTM(weight_matrix=glove_emb_matrix, in_dim=300, mem_dim=150, hidden_dim=50, num_classes=5, sparsity=False,
                               freeze=True, use_parse_tree="constituency")

    lib_dir = "/Users/manish.bansal/repos/search/treelstm.pytorch/lib"
    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])

    temp_dir = "/Users/manish.bansal/repos/search/treelstm.pytorch/temp"

    chunk_embedding = ChunkEmbedding(model, pickle_file, glove_vocab, classpath, temp_dir, 123)
    tree, mapping = chunk_embedding.constituent_parse("Mall attackers used ` less is more ' strategy", tokenize=False)
    print "done !!"
