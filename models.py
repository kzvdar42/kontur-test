import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ModuleWithWeightsInit(nn.Module):
    """`Abstract` class for torch module with weights initialization function."""

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0.01)
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Attn(ModuleWithWeightsInit):
    """Attention module for LSTM output.
    Taken & improved from https://github.com/spro/practical-pytorch/issues/56.
    """

    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.v = nn.Parameter(torch.nn.init.xavier_uniform_(torch.FloatTensor(1, self.hidden_size)))
        self.init_weights()

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.batch_score(hidden, encoder_outputs)
        softmax_out = F.softmax(attn_energies, dim=1).unsqueeze(1)
        attn_out = softmax_out.bmm(encoder_outputs.transpose(0, 1))  # (B,1,V) @ (B,V,E) -> (B,1,E)
        return attn_out.squeeze(1)

    def batch_score(self, hidden, encoder_outputs):
        length = encoder_outputs.size(0)
        batch_size = encoder_outputs.size(1)
        if self.method == 'dot':
            encoder_outputs = encoder_outputs.permute(1, 2, 0)
            energy = torch.bmm(hidden.transpose(0, 1), encoder_outputs).squeeze(1)
        elif self.method == 'general':
            energy = self.attn(encoder_outputs.view(-1, self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(hidden.transpose(0, 1), energy.permute(1, 2, 0)).squeeze(1)
        elif self.method == 'concat':
            attn_input = torch.cat((hidden.repeat(length, 1, 1), encoder_outputs), dim=2)
            energy = self.attn(attn_input.view(-1, 2 * self.hidden_size)).view(length, batch_size, self.hidden_size)
            energy = torch.bmm(self.v.repeat(batch_size, 1, 1), energy.permute(1, 2, 0)).squeeze(1)
        return energy

class PoolingModule(ModuleWithWeightsInit):
    """Module for pooling the output of LSTM."""
    
    def __init__(self, hidden_size, attn_method='dot',
                 pool_attn=False, pool_max=True, pool_min=True):
        super(PoolingModule, self).__init__()
        self.pool_attn, self.pool_max, self.pool_min = pool_attn, pool_max, pool_min
        if self.pool_attn:
            self.attn = Attn(attn_method, hidden_size)
        self.out_shape = sum([pool_attn, pool_max, pool_min]) * hidden_size
        assert self.out_shape > 0, 'At least one pooling method should be specified.'
        self.init_weights()
    
    
    def forward(self, hidden, embed_out):
        pool_results = []
        if self.pool_min:
            pool_results.append(torch.min(embed_out, 1)[0])
        if self.pool_max:
            pool_results.append(torch.max(embed_out, 1)[0])
        if self.pool_attn:
            hidden = hidden.reshape(hidden.shape[1], -1).unsqueeze(0)
            embed_out = embed_out.transpose(0, 1)
            pool_results.append(self.attn(hidden, embed_out))
        
        # Concat pooling results
        return torch.cat(pool_results, 1)




class LstmClassificator(ModuleWithWeightsInit):
    
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes, num_layers,
                 padding_idx, fc_dims=1024, dropout=0.1, **pool_kwargs):
        super(LstmClassificator, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, 
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.pooling = PoolingModule(hidden_size * 2, **pool_kwargs)
        self.fc = nn.Linear(self.pooling.out_shape, num_classes)
#         self.fc = nn.Sequential(
#             nn.Linear(self.pooling.out_shape, fc_dims),
#             nn.ReLU(),
#             nn.Linear(fc_dims, fc_dims),
#             nn.ReLU(),
#             nn.Linear(fc_dims, num_classes)
#         )
        self.init_weights()
        

    def forward(self, X):
        # Embed text
        X_embed = self.embedding(X)
        
        # LSTM
        lstm_out, (hidden, _) = self.lstm(X_embed)
        
        # Pooling
        # Get the hidden state of the last layer.
        hidden = hidden.view(self.num_layers, 2, X.shape[0], self.hidden_size)[-1]
        pooling_out = self.pooling(hidden, lstm_out)
        pooling_out = self.dropout(pooling_out)
        
        # Classification
        out = self.fc(pooling_out)
        return out

class LstmPackedClassificator(LstmClassificator):

    def forward(self, X, lens):
        # Embed text
        X_embed = self.embedding(X)
        
        # LSTM
        X_packed = pack_padded_sequence(X_embed, lens, batch_first=True, enforce_sorted=False)
        lstm_out_packed, (hidden, _) = self.lstm(X_packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)

        # Pooling
        # Get the hidden state of the second layer.
        hidden = hidden.view(self.num_layers, 2, X.shape[0], self.hidden_size)[-1]
        pooling_out = self.pooling(hidden, lstm_out)
        pooling_out = self.dropout(pooling_out)

        # Classification
        out = self.fc(pooling_out)
        return out


class DoubleLstmClassificator(ModuleWithWeightsInit):
    
    def __init__(self, vocab_size, emb_size, hidden_size, num_classes,  
                 num_layers, padding_idx, p_dropout=0.2, **pool_kwargs):
        super(LstmClassificator, self).__init__()
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=p_dropout)
        self.lstm_0 = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=p_dropout)
        self.lstm_1 = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=p_dropout)
        self.pooling = PoolingModule(hidden_size * 2, **pool_kwargs)
        # self.fc = nn.Linear(self.pooling.out_shape, num_classes)
        self.fc = nn.Sequential(
            nn.Linear(self.pooling.out_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        self.init_weights()

    def forward(self, X):
        # Embed text
        X0_embed = self.embedding(X[0])
        X1_embed = self.embedding(X[1])
        
        # Forward pass
        lstm_0_out, _ = self.lstm_0(X0_embed)
        lstm_0_out = self.pooling(lstm_0_out)

        lstm_1_out, _ = self.lstm_1(X1_embed)
        lstm_1_out = self.pooling(lstm_0_out)
        
        lstm_out = torch.cat([lstm_0_out, lstm_1_out], dim=1)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out