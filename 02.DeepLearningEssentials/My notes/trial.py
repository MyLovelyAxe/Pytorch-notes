import numpy as np

a = np.array([1,2,3,4,5])
print(a, a.shape)
a = a[:, np.newaxis]
print(a, a.shape)
print(np.zeros(5)[:, np.newaxis].shape)
c = np.loadtxt('D:/Hard Li/Python/Pytorch/PyTorch入门实战教程/03.初见深度学习/LR1-samples1.dat')
#元组可用len()返回长度
x = (2,3,4,5,6)
print(len(x))
#元组可迭代
for i in x:
    print(i)