import torch
import torch.nn as nn
class encoder(nn.Module):
    def __init__(self, hidden_size):
        super(encoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(1, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self,batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

