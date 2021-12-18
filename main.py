import argparse
import os
from preprocessing.processData import processData
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.decoder import decoder
from models.encoder import encoder

input_len=100
output_len=20

hidden_size=500

data = processData("data/raw/prices.txt", input_len, output_len, .8)
X_train,Y_train,X_test,Y_test = data.process()

batch_size=64
train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(Y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(Y_test))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

learning_rate=.0001
num_epochs=3

encoder = encoder(hidden_size)
decoder = decoder(hidden_size)

encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

criterion = nn.MSELoss()

encoder.train()
decoder.train()
for epoch in range(1, num_epochs + 1):
    for i, (xs, labels) in enumerate(train_loader):
        loss=0
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        pred = torch.zeros(output_len)
        # Perform encoding loop
        batch = xs.shape[0]
        encoder_hidden = encoder.initHidden(batch)
        for j in range(input_len):
            encoder_output, encoder_hidden = encoder(xs[:,j].reshape(-1,1,1).float(), encoder_hidden)

        # Perform decoding loop
        decoder_input=xs[:,-1].reshape(-1,1,1)
        decoder_hidden = encoder_hidden

        for j in range(0,output_len):
            decoder_output, decoder_hidden = decoder(decoder_input.float(), decoder_hidden)
            decoder_input = labels[:,j].reshape(-1,1,1)
            loss += criterion(decoder_output.flatten().flatten(), labels[:,j].float())
        # Compute loss and perform backprop

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        print(loss)

for xs, labels in test_loader:
    batch = xs.shape[0]
    encoder_hidden = encoder.initHidden(batch)
    outputs=torch.zeros(xs.shape[0],output_len+1)
    outputs[:,0]=xs[:, -1]
    for j in range(input_len):
        encoder_output, encoder_hidden = encoder(xs[:, j].reshape(-1, 1, 1).float(), encoder_hidden)

    # Perform decoding loop
    decoder_input = xs[:, -1].reshape(-1, 1, 1)
    decoder_hidden = encoder_hidden

    for j in range(0, output_len):
        decoder_output, decoder_hidden = decoder(decoder_input.float(), decoder_hidden)
        decoder_input = decoder_output
        outputs[:,j+1]=decoder_output.flatten().flatten()
    print(outputs)