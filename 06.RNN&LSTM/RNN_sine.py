'''sin(x) prediction with RNN'''

# we want to train a sin-sequence and predict single value in the sequnce for next value with RNN
# the data is selected as one sequence of sin(x) data, with 50 time epoch of x
# here we use another representation of time-sequence data instead of:
#   [seq_len, batch, feature_len]
# as: 
#   [batch, seq_len, feature_len]
# due to we only train one sequence, and sin(x) data only needs one real number to represent
# the actual data shape will be:
#   [1, 50, 1]
# and we decide to take input and target from this sequence, which cannot be the same
# input from former 49 values, target from latter 49 values
# so input and target shapes are:
#   [1, 49, 1]

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

#get parameters
def get_parameters():

    FeatureLen = 1
    HiddenSize = 16
    OutputSize = 1
    num_time_steps = 50
    lr = 0.01

    return FeatureLen, HiddenSize, OutputSize, num_time_steps, lr

# get samples
def get_samples(num_time_steps):
    # start: the start point of training or prediction in sin(x) sequence,
    # which should be a scalar and random
    # np.random.randint(3, size = 1) will return a unidimensional array with 1 element,
    # whose value is an integer in [0,3]
    # to get a scalar, we pick the 0th element of it, i.e. the element itself
    start = np.random.randint(3, size = 1)[0]
    # create 50 time epochs
    time_steps = np.linspace(start, start+10, num_time_steps)
    # data:  [num_time_steps] unidimensional
    data = np.sin(time_steps)
    # change data into a 2-dimensional array
    # data:  [num_time_steps,1]
    data = data.reshape(num_time_steps,1)
    # transfer the data into torch.tensor
    # extract the former 49 values for input
    # x:     [batch, seq_len, feature_len] ->[1,49,1]
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    # extract the latter 49 values for target
    # y:     [batch, seq_len, feature_len] ->[1,49,1]
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)
    return x,y,time_steps

# create a RNN class
class RNN(nn.Module):

    def __init__(self, FeatureLen, HiddenSize, OutputSize):

        super(RNN, self).__init__()

        self.FeatureLen = FeatureLen
        self.HiddenSize = HiddenSize
        self.OutputSize = OutputSize

        # batch_first = True:
        #   [batch, seq_len, feature_len]
        # batch_first = False:
        #   [seq_len, batch, feature_len]
        self.rnn = nn.RNN(
            input_size = FeatureLen,
            hidden_size = HiddenSize,
            num_layers = 1,
            batch_first = True
        )

        # the final output also needs an output-layer, here we choose linear layer
        self.linear = nn.Linear(in_features = HiddenSize, out_features = OutputSize)

    def forward(self, x, hidden):
        
        # get hidden_size
        # hidden_init: [num_layers,batch,hidden_len] -> [1,1,16]
        # hidden_size = torch.tensor(hidden.shape)[-1].item()
        # push data into RNN
        # x:      [batch,seq_len,feature_len]      -> [1,49,1]
        out, hidden = self.rnn(x, hidden)
        # out:    [batch,seq_len,hidden_len]      -> [1,49,16]
        # hidden: [num_layers,batch,hidden_len]   -> [1,1,16]
        # reshape the temperary 'out' to match the shape of linear-output-layer
        # out:    [batch,seq_len,hidden_len]      -> [seq_len,hidden_len]
        out = out.view(-1, self.HiddenSize)
        out = self.linear(out)
        # out:    [seq_len,hidden_len]            -> [seq_len,output_size]
        # due to output_size = 1 in this case, reshape out to match the target y with dimension increasing:
        # y:      [batch,seq_len,feature_len]      -> [1,49,1]
        # out:    [seq_len,output_size]:[50,1]     -> [batch,seq_len,feature_len]:[1,49,1]
        out = out.unsqueeze(dim = 0)
        return out, hidden

# train and predict
def run():
    
    # get parameters
    FeatureLen, HiddenSize, OutputSize, num_time_steps, lr = get_parameters()

    # instantiate model
    model = RNN(FeatureLen, HiddenSize, OutputSize)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    # hidden
    hidden = torch.zeros(1,1,HiddenSize)

    # training
    for epoch in range(6000):

        # get input and target
        x, y, time_steps = get_samples(num_time_steps)
        out, hidden = model(x, hidden)
        # hidden doesn't need to calculate gradients, so call .detach()
        hidden = hidden.detach()

        # begin to train
        loss = criterion(out, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # print process
        if epoch%300 == 0:
            print('iteration {}: loss {}'.format(epoch, loss.item()))
    
    # prediction
    # get new input for prediction
    # x_pred: [batch, seq_len, feature_len] ->[1,49,1]
    # y_pred: [batch, seq_len, feature_len] ->[1,49,1]
    x_pred, y_pred, time_steps_pred = get_samples(num_time_steps)
    predictions = []
    # use the 0th value in seq_len
    # i.e. pick a start and predict next values with this start, neglecting other values in x_pred
    # input:  [batch, 1, feature_len]       -> [1,1,1]
    input = x_pred[:,0,:]

    # offer input seq_len times one by one
    for iter in range(x_pred.shape[1]):
        # out_pred:  [batch,seq_len,feature_len] -> [1,1,1]
        # ps: the seq_len here is the input of the RNN
        # we have extract only one single value above
        # so the seq_len of input is 1
        # however, in sake of safety, still view it again
        input = input.view(1,1,1)
        out_pred, hidden = model(input, hidden)
        # use the temperary of out_pred as input for next iteration
        input = out_pred
        # out_pred:  [batch,seq_len,feature_len] -> [1,1,1]
        # .detach() declares that out_pred doesn't need to calculate gradients
        # .numpy() changed out_pred into numpy data-type
        # .ravel() flattens the out_pred, i.e. squeeze other dimension, make out_pred from [1,1,1] to [1]
        # [0] extract the only element in the unidimensional array out_pred now
        # and append it into predictions
        predictions.append(out_pred.detach().numpy().ravel()[0])

    x = x_pred.data.numpy().ravel()
    y = y_pred.data.numpy()
    plt.scatter(time_steps_pred[:-1], x.ravel(), s=90)
    plt.plot(time_steps_pred[:-1], x.ravel())

    plt.scatter(time_steps_pred[1:], predictions)
    plt.show()

if __name__ == '__main__':
    run()