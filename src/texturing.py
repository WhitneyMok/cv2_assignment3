import sys
import os
import h5py
import dlib
import math
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from tools.visualizer import *
from src.morphable_model import *
from src.pinhole_cam_model import *
from tools.supplemental_code import *
from src.latentparamEstimation import *
from src.latentparamEstimation_Model import Model
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable


def bilinear_interpolation(u, v, source_image):
    x_lower = int(u)
    x_upper = math.ceil(u)
    y_lower = int(v)
    y_upper = math.ceil(v)

    points = [[x_upper, y_upper], [x_upper, y_lower], [x_lower, y_upper], [x_lower, y_lower]]
    points = sorted(points)
    (x1, y2), (_x1, y1), (x2, _y2), (_x2, _y1) = points

    if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
        raise ValueError('points do not form a rectangle')
    if not x1 <= u <= x2 or not y2 <= v <= y1:
        raise ValueError('(x, y) not within the rectangle')

    bgr = [0, 0, 0]
    for color_channel in range(len(bgr)):
        bgr[color_channel] = (source_image[y1][x2][color_channel] * (x2 - u) * (y1 - v) +
                              source_image[y1][x1][color_channel] * (u - x1) * (y1 - v) +
                              source_image[y2][x2][color_channel] * (x2 - u) * (v - y2) +
                              source_image[y2][x1][color_channel] * (u - x1) * (v - y2)
                              ) / ((x2 - x1) * (y1 - y2) + 0.0)
    bgr.reverse()
    return bgr


def interpolate_colors(coords_2d, source_image):
    colors = []
    for iteration, i in enumerate(range(coords_2d.shape[0])):
        uv = coords_2d[i, :]
        u = uv[0]
        v = uv[1]
        rgb = bilinear_interpolation(u, v, source_image)
        colors.append(rgb)
    colors = np.asarray(colors)
    normalized_colors = colors/255
    return normalized_colors


if __name__ == "__main__":
    image = ['../Data/obama_frames/frame10502.jpg']
    original_image_2d = cv2.imread(image[0])

    load_model = True
    model_dir = '../Results/sec4/'
    model_name = 'LatentParamEst_lambdaAlpha0.01_lambdaDelta0.01_200000iter__lrate0.01_MODEL' # best model according to experiments of section 4
    model_path = model_dir + model_name
    save_dir = '../Results/sec5/'

    if load_model:
        data = read_data()
        # initialized params
        alpha, delta = sample_latent_params()
        t = nn.Parameter(Variable(torch.tensor([[0.], [0.], [-500.]], dtype=torch.double), requires_grad=True))
        omegas = degrees2radians([0, 10, 0])
        print('Loading model from {}.'.format(model_path))
        model = Model(data, alpha, delta, omegas, t)
        model.load_state_dict(torch.load(model_path))
        model.eval()

    # open the face model learned in section 4 Latent Parameter Estimation,
    base_file_name = model_name.split('MODEL')[0]
    face_vertices = represent_geometry(model.alpha, model.delta, data, draw_pointcloud=False, save_model=True, save_dir=save_dir, file_name=base_file_name+'3Dmodel.obj')
    face_vertices_homog = cartesian2homogeneous(face_vertices)
    rigidly_transformed_face = rigid_transformation(face_vertices_homog, model.omega, model.t)
    face_2Dprojection = torch.transpose(get_u_v_projection(rigidly_transformed_face), 0, 1)

    # and get the pixel colors from the source image,
    color_vertex_obama = interpolate_colors(face_2Dprojection.detach().numpy()[:, :2], original_image_2d)

    # to finally render this learned 3D face model, and project it to the camera plane.
    draw_pointcloud(face_vertices.detach().numpy(), color_vertex_obama) # 3D
    save_obj(save_dir + base_file_name + '3DmodelObama_ColorsObama.obj', face_vertices.detach().numpy(), color_vertex_obama, data['triangles'])  # save 3d Obama
    draw_pointcloud(face_2Dprojection.detach().numpy(), color_vertex_obama) # 2D projection
    save_obj(save_dir + base_file_name + '2DprojectionObama_ColorsObama.obj', face_2Dprojection.detach().numpy(), color_vertex_obama, data['triangles'])  # save 2d Obama