# tree object from stanfordnlp/treelstm
class Tree(object):
    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth


# ConstTree object from stanfordnlp/treelstm
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
