#How to write .dat file with training or testing data
import numpy as np
from Linear_Regression_Data_Generation import Generate_Linear_Regression_Data as GLRD

#Generate data
x, y = GLRD(weight = 2, bias = 3,
            gauss_noise_mu = 0, gauss_noise_sigma = 3, gauss_noise_capacity = 50,
            x_start = 0, x_end = 30, x_number = 50)
Data = np.array([x, y]).T

#Output data as .dat file
np.savetxt('D:/Hard Li/Python/Pytorch/PyTorch入门实战教程/03.初见深度学习/Preparation/LiearRegressionData.txt', Data)
