import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms

###### apply momentum to imporve training ######

batch_size=200
learning_rate=0.01
epochs=10
# momentum factor
beta = 0.78

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
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 200),
            nn.LeakyReLU(inplace=True),
            nn.Linear(200, 10),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)

        return x

device = torch.device('cuda:0')
net = MLP().to(device)
# momentum: set 'momentum' as the momentum factor 'beta'
#     the larger the beta is, the more we consider previous gradient updating
#     attention: optimizer 'Adam' doesn't need to set 'momentum'
#                because 'Adam' algrithom itself applys 'momentum'
#                check '04.Gradient&Perceptron/4 Training/AdamOptimizer.ipynb'
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=beta)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda()

        # # if we need L1-regularization, we can only manully set it:
        # L1_regularization_loss = 0
        # L1_lambda = 0.01
        # for param in net.parameters():
        #     # L1 norm: ||param||
        #     L1_regularization_loss += torch.sum(torch.abs(param))
        # logits = net(data)
        # loss = criteon(logits, target) + L1_lambda * L1_regularization_loss

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
