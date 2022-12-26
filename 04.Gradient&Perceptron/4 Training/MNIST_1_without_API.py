import  torch
import  torch.nn as nn
import  torch.nn.functional as F
import  torch.optim as optim
from    torchvision import datasets, transforms

# MNIST: hand-writting letter recognition

batch_size=200
learning_rate=0.01
epochs=10

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hardli/python/pytorch/datasets', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/hardli/python/pytorch/datasets', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)


# according to convention of torch, parameters' shape aredefinede as:
# [channel_out, channel_in]
# e.g.
# input x: [sample_num, channels, height, width] = [200,3,28,28]
# in order to train more sufficiently, flatten as [200,3,784]
# w1,b1: for 1st layer, which convert [200,3,784] to [200,3,200]
# w1 and b1 need gradients to update, so set 'requires_grad' as True
# same for other parameters
w1, b1 = torch.randn(200, 784, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
# w2,b2: for 2nd layer, which convert [200,3,200] to [200,3,200]
w2, b2 = torch.randn(200, 200, requires_grad=True),\
         torch.zeros(200, requires_grad=True)
# w3,b3: for 3rd layer, which convert [200,3,200] to [200,3,10]
# which denotes 10 propabilities for each digit predction
w3, b3 = torch.randn(10, 200, requires_grad=True),\
         torch.zeros(10, requires_grad=True)

# initialize weights w, because the defining of w were random
# bias b are unnecessary, because they were already set as zeros
torch.nn.init.kaiming_normal_(w1)
torch.nn.init.kaiming_normal_(w2)
torch.nn.init.kaiming_normal_(w3)


def forward(x):
    # b1: [200,]
    # then broadcasting
    x = x@w1.t() + b1
    x = F.relu(x)
    x = x@w2.t() + b2
    x = F.relu(x)
    x = x@w3.t() + b3
    # the F.relu(x) below can be neglected
    # because we want logits in this step
    # logits are output before activation function
    # (e.g. sigmoid, softmax, relu)
    x = F.relu(x)
    return x


optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
# softmax and cross entropy loss are integrated in torch
# usually they are called together, in order to avoid of data-mess
# this is the same with 'F.cross_entropy()'
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        # unnecessary to apply softmax
        # because softmax is already applied in nn.CrossEntropyLoss() bfore
        logits = forward(data)
        loss = criteon(logits, target)
        # clear existed gradients information in optimizer
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
        logits = forward(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
