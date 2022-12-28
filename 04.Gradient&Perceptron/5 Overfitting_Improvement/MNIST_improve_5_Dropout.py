import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms

###### apply early stop and dropout to imporve training ######

# attention:
#   torch.nn.Dropout(p = drop_prob)
#   in Pytorch, the 'prob' denotes how many neurons will be dropped
#   tf.nn.dropout(p = keep_prob)
#   in tensoflow, the 'prob' denotes how many neurons will be kept

batch_size=200
learning_rate=0.01
epochs=10
dropout_prob = 0.5

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hardli/python/pytorch/datasets', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hardli/python/pytorch/datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)

class MLP(nn.Module):

    def __init__(self):
        super(MLP, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 200),
            # how dropout works:
            #   dropout forces part of parameters not in training during epoch
            #   therefore the amount of paramters of the model will reduce
            #   which makes the model less possible to learn out-siders
            #   i.e. reduce overfitting
            # in Pytorch, just add a dropout layer between where it is necessary
            # e.g. 50% neurons 'may' be dropped
            # question:
            #   if we drop out neurons, how can we continue remain the input dimension of next layer?
            nn.Dropout(dropout_prob),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.Dropout(dropout_prob),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    # when dropout is applied, we should pay attention to this:
    #   behaviors for train and test are different
    #   we want dropout during training to reduce overfitting
    #   but we don't need that during testing
    #   so we have to manually switch 'dropout' on when training
    #   and we have to manually switch 'dropout' off when testing
    
    # swich 'dropout' on when training
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        logits = net(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # swich 'dropout' off when testing
    net.eval()

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        data, target = data.to(device), target.cuda()
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.argmax(dim=1)
        correct += pred.eq(target).float().sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
