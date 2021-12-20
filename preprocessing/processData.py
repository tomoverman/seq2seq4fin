import numpy as np
class processData():
    def __init__(self, data_path, in_len, out_len, train_frac):
        self.data_path = data_path
        self.in_len = in_len
        self.out_len = out_len
        self.train_frac = train_frac

    def process(self):
        in_len=self.in_len
        out_len=self.out_len
        data = np.loadtxt(self.data_path)
        #now split the data into portions of in_len and out_len length
        num_samples = int(np.floor(data.shape[0] / (out_len + in_len)))
        samples = np.zeros((num_samples,in_len))
        labels = np.zeros((num_samples,out_len))
        for i in range(0,num_samples):
            samples[i,:]=data[i*(in_len+out_len):i*(in_len+out_len)+in_len]
            labels[i,:]=data[i*(in_len+out_len)+in_len:i*(in_len+out_len)+in_len+out_len]
        #split into training and test sets
        num_train = int(np.floor(self.train_frac*num_samples))

        X_train=samples[0:num_train]
        X_test=samples[num_train:]

        Y_train=labels[0:num_train]
        Y_test=labels[num_train:]

        return X_train, Y_train, X_test, Y_test