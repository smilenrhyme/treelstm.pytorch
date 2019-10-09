from tqdm import tqdm

import torch
# import numpy as np
# from matplotlib.lines import Line2D
# import matplotlib.pyplot as plt

from . import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.epoch = 0

    # helper function for training
    def train(self, dataset):
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0
        batch_loss = 0.0
        indices = torch.randperm(len(dataset), dtype=torch.long, device='cpu')
        for idx in tqdm(range(len(dataset)), desc='Training epoch ' + str(self.epoch + 1) + ''):
            # print('==> Index of data tuple     : %d ' % indices[idx])
            ltree, linput, rtree, rinput, label = dataset[indices[idx]]
            # print "linput : ", dataset.vocab.convertToLabels(linput.numpy(), -1), " : ", "rinput : ", dataset.vocab.convertToLabels(rinput.numpy(), -1)
            target = utils.map_label_to_target(label, dataset.num_classes)
            linput, rinput = linput.to(self.device), rinput.to(self.device)
            target = target.to(self.device)
            output = self.model(ltree, linput, rtree, rinput)
            loss = self.criterion(output, target)
            total_loss += loss.item()
            batch_loss += loss
            if idx % self.args.batchsize == 0 and idx > 0:
                self.optimizer.zero_grad()
                batch_loss.backward()
                # self.plot_grad_flow(self.model.named_parameters())
                self.optimizer.step()
                batch_loss = 0.0
        self.epoch += 1
        return total_loss / len(dataset)

    # @staticmethod
    # def plot_grad_flow(named_parameters):
    #     """Plots the gradients flowing through different layers in the net during training.
    #     Can be used for checking for possible gradient vanishing / exploding problems.
    #
    #     Usage: Plug this function in Trainer class after loss.backwards() as
    #     "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    #     ave_grads = []
    #     max_grads = []
    #     layers = []
    #     for n, p in named_parameters:
    #         if (p.requires_grad) and ("bias" not in n):
    #             layers.append(n)
    #             ave_grads.append(p.grad.abs().mean())
    #             max_grads.append(p.grad.abs().max())
    #     plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    #     plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    #     plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    #     plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    #     plt.xlim(left=0, right=len(ave_grads))
    #     plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    #     plt.xlabel("Layers")
    #     plt.ylabel("average gradient")
    #     plt.title("Gradient flow")
    #     plt.grid(True)
    #     plt.legend([Line2D([0], [0], color="c", lw=4),
    #                 Line2D([0], [0], color="b", lw=4),
    #                 Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    # helper function for testing
    def test(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label = dataset[idx]
                target = utils.map_label_to_target(label, dataset.num_classes)
                linput, rinput = linput.to(self.device), rinput.to(self.device)
                target = target.to(self.device)
                output = self.model(ltree, linput, rtree, rinput)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss / len(dataset), predictions

    def test_sample(self, dataset):
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            predictions = torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            indices = torch.arange(1, dataset.num_classes + 1, dtype=torch.float, device='cpu')
            for idx in tqdm(range(len(dataset)), desc='Testing epoch  ' + str(self.epoch) + ''):
                ltree, linput, rtree, rinput, label = dataset[2962]
                target = utils.map_label_to_target(label, dataset.num_classes)
                linput, rinput = linput.to(self.device), rinput.to(self.device)
                target = target.to(self.device)
                output = self.model(ltree, linput, rtree, rinput, print_state=True)
                break
                loss = self.criterion(output, target)
                total_loss += loss.item()
                output = output.squeeze().to('cpu')
                predictions[idx] = torch.dot(indices, torch.exp(output))
        return total_loss / len(dataset), predictions
