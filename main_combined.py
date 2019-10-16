from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch
import torch.nn as nn
import torch.optim as optim

# IMPORT CONSTANTS
from treelstm import Constants
# NEURAL NETWORK MODULES/LAYERS
from treelstm import SimilarityTreeLSTM
# DATA HANDLING CLASSES
from treelstm import Vocab
# DATASET CLASS FOR SICK DATASET
from treelstm import SICKDataset
# METRICS CLASS FOR EVALUATION
from treelstm import Metrics
# UTILITY FUNCTIONS
from treelstm import utils
# TRAIN AND TEST HELPER FUNCTIONS
from treelstm import Trainer
# CONFIG PARSER
from config import parse_args


# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    combined_dir = os.path.join(args.data, 'combined_sick_multi/')
    logger.debug('==> Reading GLOVE embeddings !!')
    glove_vocab, glove_emb_matrix = utils.load_keyed_vectors(
        os.path.join(args.glove, 'glove.840B.300d'))
    logger.debug('==> GLOVE vocabulary size: %d ' % glove_vocab.size())

    # load SICK dataset splits
    train_file = os.path.join(args.data, 'sick_combined_multi{}.pth'.format(args.use_parse_tree))
    if os.path.isfile(train_file):
        train_dataset = torch.load(train_file)
    else:
        train_dataset = SICKDataset(combined_dir, glove_vocab, args.num_classes, args.use_parse_tree)
        torch.save(train_dataset, train_file)
    logger.debug('==> Size of train data   : %d ' % len(train_dataset))

    # initialize model, criterion/loss_function, optimizer
    model = SimilarityTreeLSTM(
        glove_emb_matrix,
        args.input_dim,
        args.mem_dim,
        args.hidden_dim,
        args.num_classes,
        args.sparse,
        args.freeze_embed,
        args.use_parse_tree)
    criterion = nn.KLDivLoss()

    model.to(device), criterion.to(device)
    if args.optim == 'adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad,
                                      model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(filter(lambda p: p.requires_grad,
                                         model.parameters()), lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad,
                                     model.parameters()), lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer, device)

    best = -float('inf')
    for epoch in range(args.epochs):
        train_loss = trainer.train(train_dataset)
        train_loss, train_pred = trainer.test(train_dataset)

        train_pearson = metrics.pearson(train_pred, train_dataset.labels)
        train_mse = metrics.mse(train_pred, train_dataset.labels)
        logger.info('==> Epoch {}, Train \tLoss: {}\tPearson: {}\tMSE: {}'.format(
            epoch, train_loss, train_pearson, train_mse))

        if best < train_pearson:
            best = train_pearson
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'pearson': train_pearson, 'mse': train_mse,
                'args': args, 'epoch': epoch
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))


if __name__ == "__main__":
    main()
