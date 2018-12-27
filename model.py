import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class Attention(nn.Module):

    def __init__(self, query_size, key_size):
        super(Attention, self).__init__()
        self.weights = nn.Parameter(torch.rand(key_size, query_size) * 0.2 - 0.1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, query, key):
        # query: (batch_size, query_size)
        # key: (batch_size, time_step, key_size)
        # value: (batch_size, time_step, value_size)
        batch_size = key.size(0)
        time_step = key.size(1)
        weights = self.weights.repeat(batch_size, 1, 1) # (batch_size, key_size, query_size)
        query = query.unsqueeze(-1)    # (batch_size, query_size, 1)
        mids = weights.matmul(query)    # (batch_size, key_size, 1)
        mids = mids.repeat(time_step, 1, 1, 1).transpose(0, 1) # (batch_size, time_step, key_size, 1)
        key = key.unsqueeze(-2)    # (batch_size, time_step, 1, key_size)
        scores = torch.tanh(key.matmul(mids).squeeze() + self.bias)   # (batch_size, time_step, 1, 1)
        scores = scores.squeeze()   # (batch_size, time_step)
        attn_weights = F.softmax(scores, dim=1)
        return attn_weights

class IAN(nn.Module):

    def __init__(self, config):
        super(IAN, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.n_class = config.n_class
        self.l2_reg = config.l2_reg
        self.dropout = config.dropout
        self.max_aspect_len = config.max_aspect_len
        self.max_context_len = config.max_context_len

        self.embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.embedding_dropout = nn.Dropout(self.dropout)
        self.aspect_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.context_lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, batch_first=True)
        self.aspect_attn = Attention(self.hidden_size, self.hidden_size)
        self.context_attn = Attention(self.hidden_size, self.hidden_size)
        self.fc = nn.Linear(self.hidden_size * 2, self.n_class)
        # self.embedding.weight.data.copy_(torch.from_numpy(config.embedding))

    def forward(self, aspect, context):
        aspect = self.embedding(aspect)
        aspect = self.embedding_dropout(aspect)
        aspect_output, _ = self.aspect_lstm(aspect)
        aspect_avg = aspect_output.mean(dim=1, keepdim=False)
        context = self.embedding(context)
        context = self.embedding_dropout(context)
        context_output, _ = self.context_lstm(context)
        context_avg = context_output.mean(dim=1, keepdim=False)
        aspect_attn = self.aspect_attn(context_avg, aspect_output).unsqueeze(1)
        aspect_features = aspect_attn.matmul(aspect_output).squeeze()
        context_attn = self.context_attn(aspect_avg, context_output).unsqueeze(1)
        context_features = context_attn.matmul(context_output).squeeze()
        features = torch.cat([aspect_features, context_features], dim=1)
        output = self.fc(features)
        return output

class IanDataset(Dataset):

    def __init__(self, path):
        data = np.load(path)
        self.aspects = torch.from_numpy(data['aspects']).long()
        self.contexts = torch.from_numpy(data['contexts']).long()
        self.labels = torch.from_numpy(data['labels']).long()
        self.len = self.labels.shape[0]

    def __getitem__(self, index):
        return self.aspects[index], self.contexts[index], self.labels[index]

    def __len__(self):
        return self.len