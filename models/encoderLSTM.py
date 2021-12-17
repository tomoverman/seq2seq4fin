import torch
import torch.nn as nn
class encoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(encoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)

    def forward(self, input):
        output, (final_hidden, final_cell) = self.lstm(input, hidden)
        return output

