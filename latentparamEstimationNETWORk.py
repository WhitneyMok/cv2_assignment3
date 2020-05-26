import sys
import os
import h5py
import dlib
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from morphable_model import *
from pinhole_cam_model import *
from supplemental_code import *
import cv2
import torch.nn as nn
from torch.autograd import Variable
import visualizer


class Network(nn.Module):
    '''
    Rsfsdfsafadfsf
    '''
    def __init__(self, input_dim, output_dim, nr_hidden_units):
        super(Network, self).__init__()
        self.alpha = nn.Parameter(Variable(torch.tensor(alpha), requires_grad=True))
        self.delta = nn.Parameter(Variable(torch.tensor(delta), requires_grad=True))
        self.omega =  nn.Parameter(Variable(torch.tensor(omega), requires_grad=True))
        self.t = nn.Parameter(Variable(torch.tensor(t), requires_grad=True))




        self.hid_units_1 = nr_hidden_units
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hid_units_1, bias=True),
            nn.ReLU(),
            nn.Linear(self.hid_units_1, output_dim, bias=True)
        )

    def forward(self, x):
        out = self.model(x)
        return out

    def predict(self, x):
        out = self.model(x)
        return out



def train(lambda_alpha):
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')






# Q4.1
image =  './Data/Bean.jpg'
image = cv2.imread(image)
landmarks = detect_landmark(image)
plt.scatter(landmarks[:, 0], landmarks[:, -1])
# plt.show()

#Q4.2



