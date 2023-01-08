'''sin(x) prediction with RNN'''

'''RNN has issue of gradient exploding. There is a solution to improve is: gradient clipping'''

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

#get samples
def get_samples(num_time_steps):

    start = np.random.randint(3, size = 1)[0]
    time_steps = np.linspace(start, start+10, num_time_steps)
    data = np.sin(time_steps)
    data = data.reshape(num_time_steps,1)
    x = torch.tensor(data[:-1]).float().view(1, num_time_steps - 1, 1)
    y = torch.tensor(data[1:]).float().view(1, num_time_steps - 1, 1)
    return x,y,time_steps

#create a RNN class
class RNN(nn.Module):

    def __init__(self, FeatureLen, HiddenSize, OutputSize):

        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size = FeatureLen,
            hidden_size = HiddenSize,
            num_layers = 1,
            batch_first = True
        )

        self.linear = nn.Linear(in_features = HiddenSize, out_features = OutputSize)

    def forward(self, x, hidden):
        
        #get hidden_size
        hidden_size = torch.tensor(hidden.shape)[-1].item()
        out, hidden = self.rnn(x, hidden)
        out = out.view(-1, hidden_size)
        out = self.linear(out)
        out = out.unsqueeze(dim = 0)
        return out, hidden

#train and predict
def run():
    
    #get parameters
    FeatureLen, HiddenSize, OutputSize, num_time_steps, lr = get_parameters()

    #instantiate model
    model = RNN(FeatureLen, HiddenSize, OutputSize)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr)
    #initialize hidden memory
    hidden = torch.zeros(1,1,HiddenSize)
    #set a threshold for gradient exploding
    threshold = 5

    #training
    for epoch in range(6000):

        #get input and target
        x, y, time_steps = get_samples(num_time_steps)
        out, hidden = model(x, hidden)
        #hidden doesn't need to calculate gradients, so call .detach()
        hidden = hidden.detach()

        #begin to train
        loss = criterion(out, y)
        model.zero_grad()
        loss.backward()
        #in order to improve the performance of gradient exploding
        #apply 'gradient clipping'
        #i.e. check if gradients are larger than 'threshold'
        #if they were, clip it under 'threshold'
        for g in model.parameters():
            #check norm of gradients
            if g.grad.norm() < threshold:
                #clip gradients whose norms are larger than threshold
                torch.nn.utils.clip_grad_norm_(g,threshold)
        optimizer.step()

        #print process
        if epoch%300 == 0:
            print('iteration {}: loss {}'.format(epoch, loss.item()))
    
    #prediction
    x_pred, y_pred, time_steps = get_samples(num_time_steps)
    predictions = []
    input = x_pred[:,0,:]

    #offer input seq_len times one by one
    for iter in range(x_pred.shape[1]):

        input = input.view(1,1,1)
        out_pred, hidden = model(input, hidden)
        input = out_pred
        predictions.append(out_pred.detach().numpy().ravel()[0])

    x = x.data.numpy().ravel()
    y = y.data.numpy()
    plt.scatter(time_steps[:-1], x.ravel(), s=90)
    plt.plot(time_steps[:-1], x.ravel())

    plt.scatter(time_steps[1:], predictions)
    plt.show()

if __name__ == '__main__':
    run()