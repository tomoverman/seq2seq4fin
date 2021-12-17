import torch
import torch.nn as nn
class encoderLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(encoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)

    def forward(self, input):
        output, (final_hidden, final_cell) = self.lstm(input)
        return final_hidden, final_cell

