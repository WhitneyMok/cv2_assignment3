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
import torch
import torch.nn as nn
import torch.optim as optim


import visualizer


#
def criterion(I, p):
    #TODO convert np to torch if needed
    # loss_lan = (1/68) *

    loss_fit = loss_lan + loss_reg
    return loss_fit


def train(model, target_img, lambda_alpha):
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    I = get_ground_truth(target_img)

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        # for i, data in enumerate(trainloader, 0):
        for i in max_iterations:
            # get the inputs
            # inputs, labels = data


            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            # outputs = net(inputs)
            p = model()
            loss = criterion(I, p)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
# #

def get_ground_truth(image, plot=False, np2torch=False):
    image = cv2.imread(image)
    landmarks = detect_landmark(image)
    if plot:
        plt.scatter(landmarks[:, 0], landmarks[:, -1])
        plt.show()

    if np2torch:
        #TODO np2torch
        pass
    return landmarks



# Q4.1
image =  './Data/Bean.jpg'
landmarks = get_ground_truth(image, plot=True)

#Q4.2
# network =
# trained_network = train(network)