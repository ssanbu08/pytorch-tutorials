
import torch
import torch.nn as nn
import config

class CharacterEmbedding():

    def __init__(self):
        super.__init__(CharacterEmbedding)
        self.embedding_dim = 100
        self.device = "cuda"
        self.hidden_size = 100
        self.char2idx = config.char2idx
        self.idx2char = config.idx2char
        self.chars = len(self.char2idx)
        # All nn. operations should be transferred to the corresponding device.
        self.dropout = nn.Dropout(0.2).to(self.device)
        self.char_embedding = nn.Embedding(self.chars, self.embedding_dim).to(self.device)
        self.char_bilstm = nn.LSTM(self.embedding_dim, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True).to(self.device)


    def forward(self, char_seq_tensor: torch.tensor, char_seq_len: torch.tensor) -> torch.tensor:
        '''
        Get the last hidden state of the LSTM
        :return:
        '''

        # Preprocess the input data to convert into format suitable
        # to be passed to the network
        # Char_embedding --> Dropout --> Char_biLSTM



