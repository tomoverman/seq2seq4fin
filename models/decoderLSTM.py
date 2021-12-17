import torch
import torch.nn as nn
class decoderLSTM(nn.Module):
    def __init__(self, hidden_size):
        super(decoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.outlayer = nn.Linear(hidden_size, 1)
        self.finalsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output_lstm, (final_hidden, final_cell) = self.lstm(input,hidden)
        output = self.finalsoftmax(self.outlayer(output_lstm))
        return output, final_hidden, final_cell