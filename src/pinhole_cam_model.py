import sys
import os
import h5py
import dlib
import math
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from src.morphable_model import *
from tools.supplemental_code import *
import tools.visualizer as visualizer
import cv2
from PIL import Image, ImageFilter


def degrees2radians(rotations_in_degrees):
    rotation_in_radians = []
    for rotation in rotations_in_degrees:
        rotation_in_radians.append(rotation * (math.pi / 180))

    rotation_in_radians = nn.Parameter(Variable(torch.tensor(rotation_in_radians), requires_grad=True))

    return rotation_in_radians


def get_ViewportMatrix(v_left=0, v_right=1600, v_top=1200, v_bottom=0):
    S = np.asarray([[(v_right-v_left)/2, 0, 0, 0],
                    [0, (v_top-v_bottom)/2, 0, 0],
                    [0, 0, 0.5, 0],
                    [0, 0, 0, 1]])

    T = np.asarray([[1, 0, 0, (v_left + v_right)/2],
                    [0, 1, 0, (v_bottom + v_top)/2],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1]])
    V = np.dot(T, S)
    return V


def get_PerspectiveProjectionMatrix(FOV=90, n=0.1, f=100):
    aspect_ratio = 1600 / 1200
    t = math.tan((FOV/2) * (math.pi/180)) * n
    r = t * aspect_ratio
    b = -t * aspect_ratio
    l = b

    P = np.asarray([[(2*n)/(r-l), 0, 0, 0],
                    [0, (2*n)/(t-b), 0, 0],
                    [(r+l)/(r-l), (t+b)/(t-b), -(f+n)/(f-n), -1],
                    [0, 0, -(2*f*n)/(f-n), 0]])
    return P


def get_pi():
    V = torch.from_numpy(get_ViewportMatrix())
    P = torch.from_numpy(get_PerspectiveProjectionMatrix())
    pi = torch.mm(V, P)
    return pi


def getRotationMatrix(omega): # omega should be an array where the elements correspond to the angle along x, y, or z axis
    #     R_x = np.array([[1, 0, 0],
    #                     [0, math.cos(omega[0]), -math.sin(omega[0])],
    #                     [0, math.sin(omega[0]), math.cos(omega[0])]
    #                     ])
    #
    #     R_y = np.array([[math.cos(omega[1]), 0, math.sin(omega[1])],
    #                     [0, 1, 0],
    #                     [-math.sin(omega[1]), 0, math.cos(omega[1])]
    #                     ])
    #
    #     R_z = np.array([[math.cos(omega[2]), -math.sin(omega[2]), 0],
    #                     [math.sin(omega[2]), math.cos(omega[2]), 0],
    #                     [0, 0, 1]
    #                     ])
    #
    #     R = np.dot(R_z, np.dot(R_y, R_x))

    # Had to do in in the following way to prevent ''losing'' gradients. #TODO if time: any better ways than all this concatenating?
    R_x_1strow = torch.tensor([1, 0, 0], dtype=torch.float,  requires_grad=True) # might not need
    R_x_2ndrow = torch.cat((torch.tensor([0], dtype=torch.float, requires_grad=True), torch.cos(omega[0]).unsqueeze(0).float(), -torch.sin(omega[0]).unsqueeze(0).float()), dim=0)
    R_x_3rdrow = torch.cat((torch.tensor([0], dtype=torch.float, requires_grad=True), torch.sin(omega[0]).unsqueeze(0).float(), torch.cos(omega[0]).unsqueeze(0).float()), dim=0)
    R_x = torch.cat((R_x_1strow, R_x_2ndrow, R_x_3rdrow), dim=0).reshape(3, 3)

    R_y_1strow = torch.cat((torch.cos(omega[1]).unsqueeze(0).float(), torch.tensor([0], dtype=torch.float, requires_grad=True), torch.sin(omega[1]).unsqueeze(0).float()), dim=0)
    R_y_2ndrow = torch.tensor([0, 1, 0], dtype=torch.float,  requires_grad=True)
    R_y_3rdrow = torch.cat((-torch.sin(omega[1]).unsqueeze(0).float(), torch.tensor([0], dtype=torch.float, requires_grad=True), torch.cos(omega[1]).unsqueeze(0).float()), dim=0)
    R_y = torch.cat((R_y_1strow, R_y_2ndrow, R_y_3rdrow), dim=0).reshape(3, 3)

    R_z_1strow = torch.cat((torch.cos(omega[2]).unsqueeze(0).float(), -torch.sin(omega[2]).unsqueeze(0).float(), torch.tensor([0], dtype=torch.float, requires_grad=True)), dim=0)
    R_z_2ndrow = torch.cat((torch.sin(omega[2]).unsqueeze(0).float(), torch.cos(omega[2]).unsqueeze(0).float(), torch.tensor([0], dtype=torch.float, requires_grad=True)), dim=0)
    R_z_3rdrow = torch.tensor([0, 0, 1], dtype=torch.float,  requires_grad=True)
    R_z = torch.cat((R_z_1strow, R_z_2ndrow, R_z_3rdrow), dim=0).reshape(3, 3)

    R = torch.mm(R_z, torch.mm(R_y, R_x)).double()
    R.requires_grad_(True)
    return R


def get_transformationmatrix(omegas, t):
    R = getRotationMatrix(omegas)
    temp = torch.cat((R, t), dim=1)
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=torch.double).view(1, 4)
    T = torch.cat((temp, bottom_row), dim=0)
    return T


def rigid_transformation(face_vertices_3d_homog_coord, omegas, xyz_translations):
    T = get_transformationmatrix(omegas, xyz_translations)
    pi = get_pi()

    face_vertices_homog_coord = torch.mm(pi, torch.mm(T, torch.transpose(face_vertices_3d_homog_coord, 0, 1)))

    return face_vertices_homog_coord


def get_u_v_projection(rigidly_transformed_face):
    d_hat = rigidly_transformed_face[-1, :]
    img = rigidly_transformed_face[:3, :] / d_hat
    return img


def cartesian2homogeneous(cartesian_points):
    homogenous_points = torch.cat((cartesian_points, torch.ones(cartesian_points.shape[0], 1, dtype=torch.double)), dim=1)
    return homogenous_points


def homogenous2cartesian(homogenous_points, axis=0):
    if axis == 0:
        return homogenous_points[:3, :].T
    elif axis == 1:
        return homogenous_points[:, :3]


def get_bfm_landmarks(face_3d, plot=False):
    f = open("../Data/Landmarks68_model2017-1_face12_nomouth.anl", "r")
    vertex_indcs_annot = [int(i.strip('\n')) for i in f.readlines()]

    transformed_face_2d = face_3d[:, :2]
    landmarks = transformed_face_2d[vertex_indcs_annot, :]

    landmarks_copy = landmarks.clone().detach().numpy()
    x = [ele for ele in landmarks_copy[:, 0].tolist()]
    y = [ele for ele in landmarks_copy[:, 1].tolist()]

    if plot:
        plt.scatter(x, y)
        # plt.savefig('./Figs/s3_Qb_{}_visualized_landmarks.png'.format(nr))
        plt.show()
    return landmarks


if __name__ == "__main__":

    # Q3.1: Rotate a face
    draw_pointcloud = True
    save_dir = '../Results/3Dmodels/'
    save = True

    print('======== Running Q3.1 ======== \n')

    data = read_data()
    alpha, delta = sample_latent_params()
    face_vertices = represent_geometry(alpha, delta, data, draw_pointcloud=draw_pointcloud) # Original face without transformation
    face_vertices_homog = cartesian2homogeneous(face_vertices)

    # (a) No translation, just rotate 10 degrees along y
    xyz_translations = nn.Parameter(Variable(torch.tensor([[0], [0], [0]], dtype=torch.double) , requires_grad=True))
    omegas = degrees2radians([0, 10, 0])
    T1 = get_transformationmatrix(omegas, xyz_translations)
    face_only_rotated_ten_degrees_pos = homogenous2cartesian(torch.mm(face_vertices_homog, T1), axis=1)

    # (b) No translation, just rotate -10 degrees along y
    omegas = degrees2radians([0, -10, 0])
    T2 = get_transformationmatrix(omegas, xyz_translations)
    face_only_rotated_ten_degrees_neg = homogenous2cartesian(torch.mm(face_vertices_homog, T2), axis=1)

    if save:
        save_obj(save_dir + 's3_Qa___ORIGINAL_.obj', face_vertices.detach().numpy(), data['color_vertex'], data['triangles'])
        save_obj(save_dir + 's3_Qa___pos_10y_rot_.obj', face_only_rotated_ten_degrees_pos.detach().numpy(), data['color_vertex'], data['triangles'])
        save_obj(save_dir + 's3_Qa___neg_10y_rot_.obj', face_only_rotated_ten_degrees_neg.detach().numpy(), data['color_vertex'], data['triangles'])


    # Q3.2: Translate, rotate face and project to 2D
    save = True
    save_dir = '../Results/2Dfrom3D/'

    print('\n======== Running Q3.2 ======== \n')

    xyz_translations = nn.Parameter(Variable(torch.tensor([[0], [0], [-500]], dtype=torch.double) , requires_grad=True))
    omegas = degrees2radians([0, 10, 0])
    rigidly_transformed_face = rigid_transformation(face_vertices_homog, omegas, xyz_translations)
    face_2Dprojection = torch.transpose(get_u_v_projection(rigidly_transformed_face), 0, 1) # project to camera plane

    if save:
        save_obj(save_dir + 's3_Qb___pos_10y_rot__-500z_trans.obj', face_2Dprojection.detach().numpy(), data['color_vertex'], data['triangles'])

    landmarks = get_bfm_landmarks(face_2Dprojection, plot=True)