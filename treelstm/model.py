import torch
import torch.nn as nn
import torch.nn.functional as F

from . import Constants


# module for childsumtreelstm
class ChildSumTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(ChildSumTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.ioux = nn.Linear(self.in_dim, 3 * self.mem_dim)
        self.iouh = nn.Linear(self.mem_dim, 3 * self.mem_dim)
        self.fx = nn.Linear(self.in_dim, self.mem_dim)
        self.fh = nn.Linear(self.mem_dim, self.mem_dim)

    def node_forward(self, inputs, child_c, child_h):
        child_h_sum = torch.sum(child_h, dim=0, keepdim=True)

        iou = self.ioux(inputs) + self.iouh(child_h_sum)
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i, o, u = F.sigmoid(i), F.sigmoid(o), F.tanh(u)

        f = F.sigmoid(
            self.fh(child_h) +
            self.fx(inputs).repeat(len(child_h), 1)
        )
        fc = torch.mul(f, child_c)

        c = torch.mul(i, u) + torch.sum(fc, dim=0, keepdim=True)
        h = torch.mul(o, F.tanh(c))
        return c, h

    def forward(self, tree, inputs):
        for idx in range(tree.num_children):
            self.forward(tree.children[idx], inputs)

        if tree.num_children == 0:
            child_c = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
            child_h = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
        else:
            child_c, child_h = zip(* map(lambda x: x.state, tree.children))
            child_c, child_h = torch.cat(child_c, dim=0), torch.cat(child_h, dim=0)

        tree.state = self.node_forward(inputs[tree.idx], child_c, child_h)
        return tree.state


class BinaryTreeLeafModule(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeLeafModule, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        self.cx = nn.Linear(self.in_dim, self.mem_dim)
        self.ox = nn.Linear(self.in_dim, self.mem_dim)

    def forward(self, inputs):
        c = self.cx(inputs)
        c = torch.reshape(c, (1, c.size(0)))
        o = F.sigmoid(self.ox(inputs))
        h = o * F.tanh(c)
        return c, h


class BinaryTreeComposer(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeComposer, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim

        def new_gate():
            lh = nn.Linear(self.mem_dim, self.mem_dim)
            rh = nn.Linear(self.mem_dim, self.mem_dim)
            return lh, rh

        self.ilh, self.irh = new_gate()
        self.lflh, self.lfrh = new_gate()
        self.rflh, self.rfrh = new_gate()
        self.ulh, self.urh = new_gate()
        self.olh, self.orh = new_gate()

    def forward(self, lc, lh, rc, rh):
        i = F.sigmoid(self.ilh(lh) + self.irh(rh))
        lf = F.sigmoid(self.lflh(lh) + self.lfrh(rh))
        rf = F.sigmoid(self.rflh(lh) + self.rfrh(rh))
        o = F.sigmoid(self.olh(lh) + self.orh(rh))
        update = F.tanh(self.ulh(lh) + self.urh(rh))
        c = i*update + lf*lc + rf*rc
        h = torch.mul(o, F.tanh(c))
        return c, h


# module for BinaryTreeLSTM
class BinaryTreeLSTM(nn.Module):
    def __init__(self, in_dim, mem_dim):
        super(BinaryTreeLSTM, self).__init__()
        self.in_dim = in_dim
        self.mem_dim = mem_dim
        self.leaf_module = BinaryTreeLeafModule(in_dim, mem_dim)
        self.composer = BinaryTreeComposer(in_dim, mem_dim)

    def forward(self, tree, inputs):
        if tree.left is None and tree.right is None:
            tree.state = self.leaf_module.forward(inputs[tree.leaf_idx])
        else:
            lc, lh = self.forward(tree.left, inputs)
            if lc is None and lh is None:
                lc = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                lh = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()

            rc, rh = self.forward(tree.right, inputs)
            if rc is None and rh is None:
                rc = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()
                rh = inputs[0].detach().new(1, self.mem_dim).fill_(0.).requires_grad_()

            tree.state = self.composer.forward(lc, lh, rc, rh)
        return tree.state


# module for distance-angle similarity
class Similarity(nn.Module):
    def __init__(self, mem_dim, hidden_dim, num_classes):
        super(Similarity, self).__init__()
        self.mem_dim = mem_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.wh = nn.Linear(2 * self.mem_dim, self.hidden_dim)
        self.wp = nn.Linear(self.hidden_dim, self.num_classes)

    def forward(self, lvec, rvec):
        mult_dist = torch.mul(lvec, rvec)
        abs_dist = torch.abs(torch.add(lvec, -rvec))
        vec_dist = torch.cat((mult_dist, abs_dist), 1)

        out = F.sigmoid(self.wh(vec_dist))
        out = F.log_softmax(self.wp(out), dim=1)
        return out


# putting the whole model together
class SimilarityTreeLSTM(nn.Module):
    def __init__(self, vocab_size, in_dim, mem_dim, hidden_dim, num_classes, sparsity, freeze, use_parse_tree):
        super(SimilarityTreeLSTM, self).__init__()
        self.emb = nn.Embedding(vocab_size, in_dim, padding_idx=Constants.PAD, sparse=sparsity)
        if freeze:
            self.emb.weight.requires_grad = False

        if use_parse_tree == 'dependency':
            self.model = ChildSumTreeLSTM(in_dim, mem_dim)
        elif use_parse_tree == 'constituency':
            self.model = BinaryTreeLSTM(in_dim, mem_dim)
        self.similarity = Similarity(mem_dim, hidden_dim, num_classes)

    def forward(self, ltree, linputs, rtree, rinputs):
        linputs = self.emb(linputs)
        rinputs = self.emb(rinputs)
        lstate, lhidden = self.model(ltree, linputs)
        rstate, rhidden = self.model(rtree, rinputs)
        output = self.similarity(lstate, rstate)
        return output
