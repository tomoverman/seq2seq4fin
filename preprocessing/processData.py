import numpy as np
class processData():
    def __init__(self, data_path, in_len, out_len, train_frac, test_frac):
        self.data_path = data_path
        self.in_len = in_len
        self.out_len = out_len
        self.train_frac = train_frac
        self.test_frac = test_frac

    def process(self):
        data = np.loadtxt(self.data_path)
        return data