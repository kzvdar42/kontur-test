import os
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import youtokentome as yttm
from functools import partial


def create_tokenizer(tokenizer_path, datasets, vocab_size, tokens, temp_file_path='tokenizer_text.temp'):
    # Load tokenizer
    if os.path.exists(tokenizer_path):
        print('Loading pretrained tokenizer...')
        tokenizer = yttm.BPE(model=tokenizer_path)
    else:
        print('Creating new tokenizer...')
        # Create the corresponding folder (if needed)
        os.makedirs(os.path.dirname(tokenizer_path), exist_ok=True)
        # Create temp file with data to train tokenizer.
        with open(temp_file_path, 'w', encoding='utf8') as out_file:
            for data in datasets:
                out_file.write('\n'.join(map(str.lower, data)))
        # Train tokenizer.
        tokenizer = yttm.BPE.train(data=temp_file_path, vocab_size=vocab_size, model=tokenizer_path,
                                n_threads=-1, **tokens)
        # Delete temp file.
        os.remove(temp_file_path)
    return tokenizer



class TextDataset(Dataset):
    """Custom dataset for train data."""

    __output_types = {'id': yttm.OutputType.ID,
                      'subword': yttm.OutputType.SUBWORD}

    def __init__(self, data, tokenizer, sample_size=None, normalize=True):
        self.data = data
        self.tokenizer = tokenizer
        self.sample_size = sample_size
        # TODO: Change `eos` token to `sep` token
        self._sep_token = self.tokenize("", eos=True)[0]
        # Find true/false indicies for future sampling.
        self.true_inds = self.data.index[self.data.answer == True]
        self.false_inds = self.data.index[self.data.answer == False]
        if normalize:
            assert not self.sample_size is None, "To normalize, sample_size must be specified."
            self.cur_set = self.normalized_sample()
        else:
            self.cur_set = self.data

    def tokenize(self, sentence, output_type='id', **kwargs):
        """Tokenize the sentence.
        :param s: the sentence to tokenize
        :param output_type: either 'id' or 'subword' for corresponding output
        :return: tokenized sentence"""
        if not isinstance(sentence, str):
            return [self.tokenize(sent, output_type, **kwargs) for sent in sentence]
        return self.tokenizer.encode(sentence.lower().strip(),
                                     output_type=self.__output_types[output_type], **kwargs)

    def decode(self, tokens):
        return [self.tokenizer.id_to_subword(token) for token in tokens]
    
    def normalized_sample(self, sample_size=None):
        if sample_size is None:
            assert not self.sample_size is None, "To normalize, sample_size must be specified."
            sample_size = self.sample_size
        sampled_inds = []
        sampled_inds.extend(np.random.choice(self.true_inds,  sample_size, replace=True))
        sampled_inds.extend(np.random.choice(self.false_inds, sample_size, replace=True))
        np.random.shuffle(sampled_inds)
        return self.data.loc[sampled_inds]

    def resample(self, **kwargs):
        self.cur_set = self.normalized_sample(**kwargs)
    
    @staticmethod
    def _my_collate(batch, pad_token):
        src, tgt = zip(*batch)
        src = [Tensor(s) for s in src]
        src = pad_sequence(src, batch_first=True, padding_value=pad_token).long()
        tgt = Tensor(tgt).long()
        return [src, tgt]

    def __len__(self):
        return len(self.cur_set)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ru  = self.tokenize(self.cur_set['ru_name'].iloc[idx],  bos=True)
        eng = self.tokenize(self.cur_set['eng_name'].iloc[idx], eos=True)
        src = ru + [self._sep_token] + eng
        tgt = self.cur_set['answer'].iloc[idx]
        return src, tgt


class TextWithLengthDataset(TextDataset):

    @staticmethod
    def _my_collate(batch, pad_token):
        src, src_lens, tgt = zip(*batch)
        src = [Tensor(s) for s in src]
        src = pad_sequence(src, batch_first=True, padding_value=pad_token).long()
        src_lens = Tensor(src_lens).long()
        tgt = Tensor(tgt).long()
        return [src, src_lens, tgt]
    
    def __getitem__(self, idx):
        src, tgt = super().__getitem__(idx)
        src_len = len(src)
        return src, src_len, tgt


class SplitTextDataset(TextDataset):

    @staticmethod
    def _my_collate(batch, pad_token):
        src, tgt = zip(*batch)
        src = list(zip(*src))
        max_len = max([len(s) for lang in src for s in lang])
        for i, lang in enumerate(src):
            lang = [Tensor(s) for s in lang]
            src[i] = pad_sequence(lang, batch_first=True, padding_value=pad_token, max_len=max_len).long()
        src = torch.stack(src, dim=0)
        tgt = Tensor(tgt).long()
        return [src, tgt]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ru  = self.tokenize(self.cur_set['ru_name'].iloc[idx],  bos=True, eos=True)
        eng = self.tokenize(self.cur_set['eng_name'].iloc[idx], bos=True, eos=True)
        tgt = self.cur_set['answer'].iloc[idx]
        return (ru, eng), tgt


class Dataloaders:
    """Custom class to handle Dataloaders for our `TextDataset`
    with the ability to pass the `resample` function."""
    
    def __init__(self, datasets, pad_token, **kwargs):
        self.dataloaders = dict()
        for k, dataset in datasets.items():
            collate_fn = collate_fn=partial(dataset._my_collate, pad_token=pad_token)
            self.dataloaders[k] = DataLoader(dataset, collate_fn=collate_fn, **kwargs)
    
    def __getitem__(self, key):
        return self.dataloaders[key]
    
    def resample(self, keys=None, **kwargs):
        if not keys:
            keys = self.dataloaders
        for k in keys:
            self.dataloaders[k].dataset.resample(**kwargs)



def pad_sequence(sequences, batch_first=False, padding_value=0, max_len=0):
    r"""Pad a list of variable length Tensors with ``padding_value``
    Taken from the Pytorch library, modified with the `max_len` argument.
    """

    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(max_len, max([s.size(0) for s in sequences]))
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor

    return out_tensor

