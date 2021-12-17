import argparse
import os
from preprocessing.processData import processData
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.decoderLSTM import decoderLSTM
from models.encoderLSTM import encoderLSTM

input_len=100
output_len=20

hidden_size=100

data = processData("data/raw/prices.txt", input_len, output_len, .8)
X_train,Y_train,X_test,Y_test = data.process()

batch_size=64
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

learning_rate=0.5
num_epochs=5

encoder = encoderLSTM(hidden_size)
decoder = decoderLSTM(hidden_size)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

encoder.train()
decoder.train()
for epoch in range(1, num_epochs + 1):
    for i, (xs, labels) in enumerate(train_loader):
        loss=0

        # Perform encoding forward pass
        encoded_hidden, encoder_cell = encoder(xs.reshape(xs.shape[0],-1,1).float())

        # Perform decoding loop
        input=xs[:,-1].reshape(-1,1,1)
        hidden = encoded_hidden
        cell = encoder_cell

        pred = []

        for j in range(0,output_len):
            hidden = hidden.reshape(1,input.shape[0],-1)
            output, out_hidden, out_cell = decoder(input.float(),(hidden, cell))
            loss += criterion(output.flatten(1), labels[:,j].reshape(-1,1).float())
            input=output
            hidden=out_hidden
            cell=out_cell
        print(output)

        # Compute loss and perform backprop
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        print(pred)
        print(loss)