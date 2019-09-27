from tree import Tree
import codecs
from tqdm import tqdm


class ConstTree(object):
    def __init__(self):
        self.left = None
        self.right = None

    def size(self):
        self.size = 1
        if self.left is not None:
            self.size += self.left.size()
        if self.right is not None:
            self.size += self.right.size()
        return self.size

    def set_spans(self):
        if self.word is not None:
            self.span = self.word
            return self.span

        self.span = self.left.set_spans()
        if self.right is not None:
            self.span += ' ' + self.right.set_spans()
        return self.span

    def get_labels(self, spans, labels, dictionary):
        if self.span in dictionary:
            spans[self.idx] = self.span
            labels[self.idx] = dictionary[self.span]
        if self.left is not None:
            self.left.get_labels(spans, labels, dictionary)
        if self.right is not None:
            self.right.get_labels(spans, labels, dictionary)


def __read_chunks(fpath):
    with codecs.open(fpath, mode="rt", encoding="utf-8") as fp:
        chunks = fp.readlines()
        chunks = [line.split("\n")[0].strip().lower() for line in chunks]
        chunks = [l[1:-1].split("] [") for l in chunks]
        chunks = [[e.strip() for e in l] for l in chunks]

        chunks = [[e.split() for ix, e in enumerate(l)] for l in chunks]
        return chunks


def __read_trees(fpath):
    with open(fpath, 'r') as f:
        trees = [__read_dep_tree(line) for line in tqdm(f.readlines())]
    return trees


def __read_dep_tree(line):
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


class DepTree(object):
    def __init__(self):
        self.children = []
        self.lo, self.hi = None, None

    def size(self):
        self.size = 1
        for c in self.children:
            self.size += c.size()
        return self.size

    def set_spans(self, words):
        self.lo, self.hi = self.idx, self.idx + 1
        if len(self.children) == 0:
            self.span = words[self.idx]
            return
        for c in self.children:
            c.set_spans(words)
            self.lo = min(self.lo, c.lo)
            self.hi = max(self.hi, c.hi)
        self.span = ' '.join(words[self.lo : self.hi])

    def get_labels(self, spans, labels, dictionary):
        if self.span in dictionary:
            spans[self.idx] = self.span
            labels[self.idx] = dictionary[self.span]
        for c in self.children:
            c.get_labels(spans, labels, dictionary)


def __read_dep_tree_1(parents):
    trees = []
    root = None
    size = len(parents)
    for i in xrange(size):
        trees.append(None)

    for i in xrange(size):
        if not trees[i]:
            idx = i
            prev = None
            prev_idx = None
            while True:
                tree = DepTree()
                parent = parents[idx] - 1

                # node is not in tree
                if parent == -2:
                    break

                tree.parent, tree.idx = parent, idx
                if prev is not None:
                    tree.children.append(prev)
                trees[idx] = tree
                if parent >= 0 and trees[parent] is not None:
                    trees[parent].children.append(tree)
                    break
                elif parent == -1:
                    root = tree
                    break
                else:
                    prev = tree
                    prev_idx = idx
                    idx = parent
    return root


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


def inOrder(root, span_list):
    if root:
        inOrder(root.left, span_list)
        span_list.append(root.span.lower())
        inOrder(root.right, span_list)


def __make_constituent_spans(parents_path, tokens_path, ists_chunks):
    const_trees, toks = [], []

    with open(tokens_path, 'r') as tp:
        for line in tp:
            toks.append(line.strip().split())

    parents = []
    with open(parents_path, 'r') as pp:
        for line in pp:
            parents.append(map(int, line.split()))

    for i in xrange(len(toks)):
        const_trees.append(__read_constituency_tree(parents[i], toks[i]))

    total_chunks = 0
    match_total = 0
    match_partial = 0

    for i in xrange(len(ists_chunks)):
        chunk = ists_chunks[i]
        tree = const_trees[i]
        tree.set_spans()
        span_list = []
        inOrder(tree, span_list)

        chunk = [' '.join(element) for element in chunk]
        total_chunks += len(chunk)

        res = set(chunk).intersection(set(span_list))
        if len(res) == len(chunk):
            match_total += 1

        match_partial += len(res)

    print "Sentence level match : ", (float(match_total) / len(ists_chunks)) * 100.0
    print "Chunk level match : ", (float(match_partial) / total_chunks) * 100.0


if __name__ == '__main__':
    ists_chunks = __read_chunks("/Users/manish.bansal/repos/search/treelstm.pytorch/ists_headlines_sent1_chunk.txt")
    __make_constituent_spans("/Users/manish.bansal/repos/search/treelstm.pytorch/data/sick/train/a.cparents", "/Users/manish.bansal/repos/search/treelstm.pytorch/data/sick/train/a.toks", ists_chunks)
    # __read_constituency_tree([9,11,11,12,13,14,15,15,10,0,9,10,12,13,14], ['The', 'small', 'boy',  'is', 'standing',  'in',  'the',  'forest'])
