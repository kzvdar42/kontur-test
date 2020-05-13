import logging
import numpy as np
from tqdm.notebook import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from typing import Dict


def accuracy(y_pred: Tensor, y_true: Tensor):
    """Compute accuracy.

    :param y_pred: model output
    :param y_true: target values
    :return: accuracy score
    """
    y_pred = y_pred.topk(1).indices.squeeze()
    return np.mean((y_pred == y_true).cpu().numpy())


def run_model(model: nn.Module, data_iterator: DataLoader, is_train_phase: bool,
              criterion, optimizer, desc: str =""):
    """Run the model through the given data.

    :param model: model to run
    :param data_iterator: iterator for data
    :param is_train_phase: if `True` run model in train mode
    :param criterion: criterion for loss
    :param optimizer: optimizer for the model
    :param desc: description for the status printing
    :returns: dict of accuracy ('acc'), and loss ('loss')
    """
    # Get device from the model
    device = next(model.parameters()).get_device()
    # Put the model in corresponding mode.
    if is_train_phase:
        model.train()
    else:
        model.eval()
    total_loss, total_acc = 0.0, 0.0
    pbar = tqdm(total=len(data_iterator), desc=desc, position=0, leave=True)
    for i, data in enumerate(data_iterator):
        for j, tensor in enumerate(data):
            data[j] = tensor.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled(is_train_phase):
            out = model(*data[:-1])
            loss = criterion(out, data[-1])

            if is_train_phase:
                loss.backward()
                optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy(out, data[-1])
        pbar.update(1)
        pbar.set_description(desc + f'- loss: {total_loss / (i + 1):7.4}'
                                  + f'; acc: {total_acc / (i + 1):7.4}')
    pbar.close()
    return {'acc':  total_acc  / (i + 1),
            'loss': total_loss / (i + 1)}


def save_model(model: nn.Module, stats: Dict, model_save_path: str):
    """Save model in provided path."""
    tqdm.write('Saving model...')
    try:
        torch.save(model, model_save_path)
        tqdm.write('Saved successfully')
    except FileNotFoundError:
        tqdm.write('Error during saving!')


def train_model(model, n_epochs, data_iterators, criterion, optimizer, 
                scheduler=None, stats=None, model_save_path=None):
    """Train the model.
    """
    if stats is None:
        stats = {'train':{'acc':[], 'loss':[]},
                 'val':  {'acc':[], 'loss':[]}}
    start_epoch_num = len(stats['train']['loss'])
    best_loss = min(stats['val']['loss']) if stats['val']['loss'] else 0
    
    try:
        for epoch in range(start_epoch_num, start_epoch_num + n_epochs):
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            tqdm.write(f'------------ Epoch {epoch}; lr: {lr:.5f} ------------')
            # Resample data from datasets
            tqdm.write('Resample data from datasets...')
            data_iterators.resample(['train'])
            for phase in ['train', 'val']:
                desc = f'{phase.title()} Epoch #{epoch} '
                epoch_stats = run_model(model, data_iterators[phase], phase == 'train',
                                        criterion, optimizer, desc)
                # Save and print stats
                for name, val in epoch_stats.items():
                    stats[phase][name].append(val)
                    tqdm.write(f'{phase.title()} {name.title()}: '
                               + ' -> '.join(map(lambda x: f"{x:.4}",
                                                 stats[phase][name][-2:])))

            # If the best validation loss, save the model.
            if best_loss is None or stats['val']['loss'][-1] < best_loss:
                best_loss = stats['val']['loss'][-1]
                tqdm.write('Smallest val loss')
                if model_save_path:
                    save_model(model, stats, model_save_path)
            if scheduler:
                scheduler.step()
    except (Exception, KeyboardInterrupt) as e:
        if isinstance(e, KeyboardInterrupt):
            tqdm.write('Training interrupted...')
        else:
            logging.error('Error at %s', 'train', exc_info=e)
        tqdm.write('Returning with current progress')
    return stats
