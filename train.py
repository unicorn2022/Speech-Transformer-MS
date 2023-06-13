import numpy as np
# from torch import nn
from tqdm import tqdm

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.nn.optim import Adam
from mindspore.dataset import transforms as T
from mindspore.dataset import GeneratorDataset
from mindspore.dataset.vision import Inter
from mindspore.train.summary import SummaryRecord
import mindspore.dataset as ds
import mindspore.common.dtype as mstype

from config import cfg, device
from data_gen import AiShellDataset
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

summary_record = SummaryRecord(log_dir="./logs")

def train_net(args):
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(cfg.sos_id, cfg.eos_id, cfg.vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)
        # print(model)
        # model = nn.DataParallel(model)

        # optimizer
        optimizer = TransformerOptimizer(
            mindspore.nn.Adam(model.trainable_params(), learning_rate=args.lr, beta1=0.9, beta2=0.98, eps=1e-09))

    else:
        checkpoint = load_checkpoint(checkpoint)
        load_param_into_net(model, checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        optimizer = TransformerOptimizer(
            mindspore.nn.Adam(model.trainable_params(), learning_rate=args.lr, beta1=0.9, beta2=0.98, eps=1e-09))

    logger = get_logger()

    #Move to GPU, if available
    model = model.to(cfg.device)
    
    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    train_loader = ds.GeneratorDataset(train_dataset, column_names=["feature", "trn"], shuffle=True)
    train_loader = train_loader.batch(batch_size=args.batch_size, drop_remainder=True)
    train_loader = train_loader.map(input_columns=["feature"], operations=[T.PadEnd([cfg.input_dim], 0)])
    train_loader = train_loader.map(input_columns=["feature"], operations=[T.TypeCast(mstype.float32)])
    
    valid_dataset = AiShellDataset(args, 'dev')
    valid_loader = ds.GeneratorDataset(valid_dataset, column_names=["feature", "trn"], shuffle=False)
    valid_loader = valid_loader.batch(args.batch_size, drop_remainder=True)
    valid_loader = valid_loader.map(input_columns=["feature"], operations=[T.PadEnd([cfg.input_dim], 0)])
    valid_loader = valid_loader.map(input_columns=["feature"], operations=[T.TypeCast(mstype.float32)])

    # Epochs
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger)
        summary_record.record(epoch, train_loss, "train_loss")

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        valid_loss = valid(valid_loader=valid_loader,
                           model=model,
                           logger=logger)
        summary_record.record(epoch, valid_loss, "valid_loss")

        # Check if there was an improvement
        is_best = valid_loss < best_loss
        best_loss = min(valid_loss, best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger):
    model.set_train(True)  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()

    # Batches
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        # Forward prop.
        pred, gold = model(padded_input, input_lengths, padded_target)
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Back prop.
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % cfg.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg


def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()

    # Batches
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, input_lengths = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        input_lengths = input_lengths.to(device)

        with context.no_grad():
            # Forward prop.
            pred, gold = model(padded_input, input_lengths, padded_target)
            loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Keep track of metrics
        losses.update(loss.item())

    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()

# import numpy as np
# import torch
# from torch.utils.tensorboard import SummaryWriter
# # from torch import nn
# from tqdm import tqdm

# from config import device, print_freq, vocab_size, sos_id, eos_id
# from data_gen import AiShellDataset, pad_collate
# from transformer.decoder import Decoder
# from transformer.encoder import Encoder
# from transformer.loss import cal_performance
# from transformer.optimizer import TransformerOptimizer
# from transformer.transformer import Transformer
# from utils import parse_args, save_checkpoint, AverageMeter, get_logger


# def train_net(args):
#     torch.manual_seed(7)
#     np.random.seed(7)
#     checkpoint = args.checkpoint
#     start_epoch = 0
#     best_loss = float('inf')
#     writer = SummaryWriter()
#     epochs_since_improvement = 0

#     # Initialize / load checkpoint
#     if checkpoint is None:
#         # model
#         encoder = Encoder(args.d_input * args.LFR_m, args.n_layers_enc, args.n_head,
#                           args.d_k, args.d_v, args.d_model, args.d_inner,
#                           dropout=args.dropout, pe_maxlen=args.pe_maxlen)
#         decoder = Decoder(sos_id, eos_id, vocab_size,
#                           args.d_word_vec, args.n_layers_dec, args.n_head,
#                           args.d_k, args.d_v, args.d_model, args.d_inner,
#                           dropout=args.dropout,
#                           tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
#                           pe_maxlen=args.pe_maxlen)
#         model = Transformer(encoder, decoder)
#         # print(model)
#         # model = nn.DataParallel(model)

#         # optimizer
#         optimizer = TransformerOptimizer(
#             torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

#     else:
#         checkpoint = torch.load(checkpoint)
#         start_epoch = checkpoint['epoch'] + 1
#         epochs_since_improvement = checkpoint['epochs_since_improvement']
#         model = checkpoint['model']
#         optimizer = checkpoint['optimizer']

#     logger = get_logger()

#     # Move to GPU, if available
#     model = model.to(device)

#     # Custom dataloaders
#     train_dataset = AiShellDataset(args, 'train')
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
#                                                pin_memory=True, shuffle=True, num_workers=args.num_workers)
#     valid_dataset = AiShellDataset(args, 'dev')
#     valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate,
#                                                pin_memory=True, shuffle=False, num_workers=args.num_workers)

#     # Epochs
#     for epoch in range(start_epoch, args.epochs):
#         # One epoch's training
#         train_loss = train(train_loader=train_loader,
#                            model=model,
#                            optimizer=optimizer,
#                            epoch=epoch,
#                            logger=logger)
#         writer.add_scalar('model/train_loss', train_loss, epoch)

#         lr = optimizer.lr
#         print('\nLearning rate: {}'.format(lr))
#         writer.add_scalar('model/learning_rate', lr, epoch)
#         step_num = optimizer.step_num
#         print('Step num: {}\n'.format(step_num))

#         # One epoch's validation
#         valid_loss = valid(valid_loader=valid_loader,
#                            model=model,
#                            logger=logger)
#         writer.add_scalar('model/valid_loss', valid_loss, epoch)

#         # Check if there was an improvement
#         is_best = valid_loss < best_loss
#         best_loss = min(valid_loss, best_loss)
#         if not is_best:
#             epochs_since_improvement += 1
#             print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
#         else:
#             epochs_since_improvement = 0

#         # Save checkpoint
#         save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


# def train(train_loader, model, optimizer, epoch, logger):
#     model.train()  # train mode (dropout and batchnorm is used)

#     losses = AverageMeter()

#     # Batches
#     for i, (data) in enumerate(train_loader):
#         # Move to GPU, if available
#         padded_input, padded_target, input_lengths = data
#         padded_input = padded_input.to(device)
#         padded_target = padded_target.to(device)
#         input_lengths = input_lengths.to(device)

#         # Forward prop.
#         pred, gold = model(padded_input, input_lengths, padded_target)
#         loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

#         # Back prop.
#         optimizer.zero_grad()
#         loss.backward()

#         # Update weights
#         optimizer.step()

#         # Keep track of metrics
#         losses.update(loss.item())

#         # Print status
#         if i % print_freq == 0:
#             logger.info('Epoch: [{0}][{1}/{2}]\t'
#                         'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

#     return losses.avg


# def valid(valid_loader, model, logger):
#     model.eval()

#     losses = AverageMeter()

#     # Batches
#     for data in tqdm(valid_loader):
#         # Move to GPU, if available
#         padded_input, padded_target, input_lengths = data
#         padded_input = padded_input.to(device)
#         padded_target = padded_target.to(device)
#         input_lengths = input_lengths.to(device)

#         with torch.no_grad():
#             # Forward prop.
#             pred, gold = model(padded_input, input_lengths, padded_target)
#             loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

#         # Keep track of metrics
#         losses.update(loss.item())

#     # Print status
#     logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

#     return losses.avg


# def main():
#     global args
#     args = parse_args()
#     train_net(args)


# if __name__ == '__main__':
#     main()