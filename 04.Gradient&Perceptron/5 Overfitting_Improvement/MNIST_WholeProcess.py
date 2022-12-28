'''Mnist Dataset Practice: L2 Regularization'''

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

'''get training loader and validation loader'''
def training_and_valid_loader(Batch_Size):
    #transforms.Compose() combine several transformation together
    #transforms.Normalize((mean[1],...,mean[n]), (std[1],..,std[n]))
    #normalize a tensor image with mean and standard deviation for each n channel
    #mnist data only has one channel (gray-scale), so as below
    data = datasets.MNIST(
        root = '/home/hardli/python/pytorch/datasets',
        train = True,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
    )
    #split the data into train dataset and validation dataset
    train_data, valid_data = torch.utils.data.random_split(data, [50000,10000])
    #set up train loader
    train_loader = DataLoader(train_data, Batch_Size, shuffle = True)
    #set up validation loader
    valid_loader = DataLoader(valid_data, Batch_Size, shuffle = True)
    return train_loader, valid_loader

'''get testing loader containing testing data'''
def testing_loader(Batch_Size):
    test_data = datasets.MNIST(
        root = '/home/hardli/python/pytorch/datasets',
        train = False,
        download = True,
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])
    )
    test_loader = DataLoader(test_data, Batch_Size, shuffle = True)
    return test_loader

'''create a multi-layer perceptron class'''
class MLP(nn.Module):
    #MLP is inherited from nn.Module
    def __init__(self):
        super(MLP, self).__init__()

        #nn.Sequential() can combine all models as a cascading one
        #e.g. the output of first nn.Linear will be pushed as input into first nn.ReLU
        #the output of first nn.ReLU will be as input into second nn.Linear 
        self.model = nn.Sequential(
            nn.Linear(784,200),
            nn.LeakyReLU(inplace = True),
            nn.Linear(200,200),
            nn.LeakyReLU(inplace = True),
            nn.Linear(200,10),
            nn.LeakyReLU(inplace = True)
        )
    #forward calculation
    def forward(self,x):
        x = self.model(x)
        return x

def run():
    Batch_Size = 200
    learning_rate = 0.001
    epochs = 10
    #get train loader and test loader
    train_loader, valid_loader = training_and_valid_loader(Batch_Size)
    test_loader = testing_loader(Batch_Size)

    device = torch.device('cuda:0')
    #create an object of class MLP
    #.to(device) moves tensors from cpu to gpu
    net = MLP().to(device)
    #create optimizer and loss function
    #SGD: Stochastic Gradient Descent
    #choose 'cross entropy' loss function here
    #use L2-regularization tricl by setting para of 'weight_decay', i.e. lamda = 0.01
    #i.e. add a part in Loss Function: 0.5 * ( lamda * sum(||paras|| ^ 2 ))
    optimizer = torch.optim.SGD(net.parameters(), lr = learning_rate, weight_decay = 0.01)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    #the same dateset should be learned for several 'epoches'
    for epoch in range(epochs):

        for batch_idx, (data, target) in enumerate(train_loader):
            #data in train_loader is of [200,1,28,28]
            #so the '-1' dim in data.view() will have length of
            #(200*1*28*28) / (28*28) = 200
            #i.e. data.view() will be of [200, 784]
            #which is just the dimension of requesed input-dim of forward()
            data = data.view(-1, 28*28)
            #.cuda() does the same function with .to(device)
            data, target = data.to(device), target.cuda()
            #data before activation function
            logits = net(data)
            #the nn.CrossEntropyLoss() has already contained operation of softmax
            #do not call .softmax() again
            #logits is of [200,10], i.e. [pic_num, classes]
            #target is of [200], i.e. [classes]
            #loss is scalar, because para of 'reduction' has default: 'mean', i.e. return weighted mean
            #attention:
            #input of torch.nn.CrossEntropyLoss() objects is Input (1st para) and Target (2nd para)
            #Input: with shape of [N,C], like[pic_num, classes]
            #Target: if target is label, then with shape of [C], like [classes]
            #        if target is possibility, then with same shape of Input
            #        here our targets are labels of numbers in [0,9], so with shape of [C]
            #loss is a torch.FloatTensor with 0-dimension, i.e. scalar, containing a float value
            loss = criterion(logits, target)
            #set gradients info to 0
            optimizer.zero_grad()
            #take gradients of all weights and bias
            loss.backward()
            #update all weights and bias
            optimizer.step()
            

            #there are 250 batches in dataloader, so show process info 5 times
            if batch_idx % 50 == 0:
                #data is a 4-dimensional tensor containing 200pics, 1 channel, 28x28 pixels, i.e. [200,1,28,28]
                #len(data) return the length of highest dimension, i.e. return 200 from 0th-dim
                #batch_idx represents the batches that have already been trained
                #so 'batch_idx * len(data)' means the pics that have been trained for now
                #'len(train_loader.dataset)' return the length of original dataset of loader, which is 50000
                #'100. * batch_idx /len(train_loader)' means the ratio of already been trained data to all data
                #'Loss.item()' show current loss
                #{:.6f} means keep 6 digit after komma in fload data
                #loss.item() extracts out the float value in the 0-dimensional torch.FloatTensor loss
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx /len(train_loader),
                    loss.item()))

        #use validation data to test the overfitting in every epoch
        test_loss = 0
        correct = 0
        for data, target in valid_loader:
            data = data.view(-1, 28*28)
            data, target = data.to(device), target.cuda()
            logits = net(data)
            #the sum of all losses in test_loader
            test_loss += criterion(logits, target).item()

            #logits.data return the data in logits, they are the same
            #.max(1) means take max along dim = 1
            #because logit.data is of [200,10], i.e. 200 pics, 10 possibilities to numbers in [0,9]
            #so it takes maximums of every row, i.e. pick the highest possibility as predicted label
            #logits.data.max[1] return an object, <class 'torch.return_types.max'>
            #which has length of 2
            #0th-element: type: 'torch.FloatTensor', size: [200], content: the max values
            #1st-element: type: 'torch.LongTensor', size: [200], content: the indices of maximums
            #so 'pred' get the indices of maximums, i.e. labels/numbers for predicted handwritting
            pred = logits.data.max(dim = 1)[1]
            #calculate how many predictions are equal to target label
            #.sum() return the correct numbers within one batch
            #and '+=' gets the number of all correct predictions along all batches(whole loader)
            correct += pred.eq(target.data).sum()

        #the average loss of all losses
        test_loss /= len(valid_loader.dataset)

        #test_loss: the average loss of all losses
        #correct:   number of all correct predictions in whole dataloader
        #len(test_loader.dataset): 1000 testing sample in testing dataset
        #100. * correct / len(test_loader.dataset): correction percentage
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss,
            correct,
            len(valid_loader.dataset),
            100. * correct / len(valid_loader.dataset)))

    #use test data to test the whole effect of trainning
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criterion(logits, target).item()

        pred = logits.data.max(dim = 1)[1]
        correct += pred.eq(target.data).sum()

    #the average loss of all losses
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':
    run()