import torch
import torch.nn as nn
class decoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(decoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.outlayer = nn.Linear(hidden_size, output_size)
        self.finalsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output, (final_hidden, final_cell) = self.lstm(input)
        output = self.finalsoftmax(self.outlayer(output))
        return output