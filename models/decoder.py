import torch
import torch.nn as nn
class decoder(nn.Module):
    def __init__(self, hidden_size):
        super(decoder, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(1, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, input, hidden):
        output, hidden = self.gru(input, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size)