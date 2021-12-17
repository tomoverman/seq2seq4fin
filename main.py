import argparse
import os
from preprocessing.processData import processData
from utils import train_model
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

data = processData("data/raw/prices.txt", 100, 20, .8, .2)
print(data.process())