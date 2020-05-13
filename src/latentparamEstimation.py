import sys
import os
import h5py
import dlib
import math
import numpy as np
import random
import matplotlib.pyplot as plt
from src.morphable_model import *
from src.pinhole_cam_model import *
from src.latentparamEstimation_Model import Model
import cv2
import torch
import pickle as pkl
import torch.nn as nn
import torch.optim as optim
from tools.supplemental_code import *
import tools.visualizer as visualizer


def get_ground_truth(image, plot=False):
    image = cv2.imread(image[0])
    landmarks = detect_landmark(image)
    if plot:
        plt.scatter(landmarks[:, 0], landmarks[:, -1])
        plt.show()
    return torch.from_numpy(landmarks).float()


def criterion(I, p, alpha, delta, lambda_alpha, lambda_delta):
    loss_lan = (1/68) * torch.norm(p.double() - I.double())**2
    loss_reg = lambda_alpha * (torch.sum(alpha**2)) + lambda_delta * (torch.sum(delta**2))
    loss_fit = loss_lan + loss_reg
    return loss_fit


def test(model, I, lambda_alpha, lambda_delta):
    ''' Test model performance '''
    with torch.no_grad():
        model.eval()
        p = model()
        loss = criterion(I, p, model.alpha, model.delta, lambda_alpha, lambda_delta)
    return p, loss


def train(model, I, lambda_alpha, lambda_delta, l_rate, max_iterations=1):
    optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)
    I.requires_grad_(True)

    stop = False
    training_losses = []

    i = 0
    while not stop:
        i += 1
        # print('=========        Iteration {}   ========='.format(i))
        optimizer.zero_grad()

        # forward + backward + optimize
        p = model()
        loss = criterion(I, p, model.alpha, model.delta, lambda_alpha, lambda_delta)
        # print(loss.item())
        training_losses.append(loss.item())
        loss.backward()
        optimizer.step()

        # print statistics
        if i % 1000 == 0:
            print('Iteration %d,  loss: %.3f' % (i + 1, loss.item()))
        # running_loss += loss.item()

        # check if time to stop
        if i == max_iterations: #TODO if time: implement early stopping
            stop = True
    print('Finished/terminated training in iteration {}, where the training loss was {} '.format(i, loss.item()))
    return model, training_losses


def visualize_results(predicted_landmarks, ground_truth_landmarks, savelandmarks=False):
    plt.scatter(ground_truth_landmarks[:, 0], ground_truth_landmarks[:, -1], label='Target landmarks')
    plt.scatter(predicted_landmarks[:, 0], predicted_landmarks[:, -1], label='Model landmarks')
    plt.legend()
    if savelandmarks:
        plt.savefig('../Results/sec4/LatentParamEst_lambdaAlpha{}_lambdaDelta{}_{}iter__lrate{}_LANDMARKS.png'.format(lambda_alpha, lambda_delta, max_iter, l_rate))
    plt.show()


def get_frames(nr_of_frames):
    if nr_of_frames == 1:
        frames = ['../Data/obama_frames/frame10502.jpg']
    else:
        dir = '../Data/obama_frames/'
        frames = []
        start_frame = 10502 - nr_of_frames
        for frame_nr in range(start_frame + 1, 10502 + 1):
            frames.append(dir + "frame%d.jpg" % frame_nr)
    return frames


if __name__ == "__main__":

    nr_of_frames = 1
    image = get_frames(nr_of_frames)

    # Q4.1 just show that landmarks can be extracted from an image of a face
    print('======== Running Q4.1 ======== \n')
    landmarks = get_ground_truth(image, plot=True)

    # Q4.2 use Energy Minimization to estimate alpha, delta, omega, t for a face given a 2D image of it
    save_model = False

    print('\n======== Running Q4.2 ======== \n')
    data = read_data()
    I = get_ground_truth(image)

    # initialize params
    alpha, delta = sample_latent_params()
    t = nn.Parameter(Variable(torch.tensor([[0.], [0.], [-500.]], dtype=torch.double) , requires_grad=True))
    omegas = degrees2radians([0., 10., 0.])

    # hyperparams
    l_rate = 0.01
    max_iter = 5#  199999 #200000 #55000
    param_list = [(0.01, 0.01)]#[(0.04, 0.01), (0.07, 0.01), (0.01, 0.01)]

    for params in param_list:
        lambda_alpha = params[0]
        lambda_delta = params[1]
        print('\n\n\n ========================== Lambda alpha: {}      Lambda delta: {}   ====================='.format(lambda_alpha, lambda_delta))
        model = Model(data, alpha, delta, omegas, t)
        trained_network, training_losses = train(model, I, lambda_alpha, lambda_delta, l_rate, max_iterations=max_iter)

        if save_model:
            torch.save(trained_network.state_dict(), './sec4/LatentParamEst_lambdaAlpha{}_lambdaDelta{}_{}iter__lrate{}_MODEL'.format(lambda_alpha, lambda_delta, max_iter, l_rate))

        p, test_loss = test(trained_network, I, lambda_alpha, lambda_delta)
        print('Final training loss: {},        Test loss: {}'.format(training_losses[-1], test_loss))
        visualize_results(p.detach().numpy(), I.detach().numpy(), savelandmarks=False)

        pkl.dump(training_losses, open('../Results/sec4/LatentParamEst_lambdaAlpha{}_lambdaDelta{}_{}iter__lrate{}__'.format(lambda_alpha, lambda_delta, max_iter, l_rate)
                      + 'TrainingLoss.pkl', 'wb'))
        file = open(('../Results/sec4/LatentParamEst_lambdaAlpha{}_lambdaDelta{}_{}iter__lrate{}__'.format(lambda_alpha, lambda_delta, max_iter, l_rate)
                      + 'TestLoss.pkl' + '.txt'), 'a+')
        file.write('Test loss: {}    (and training loss: {}\nomega init: {}\nt init: {} )'.format(test_loss, training_losses[-1], omegas, t))
        file.close()