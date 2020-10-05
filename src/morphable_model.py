import sys
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import random
from tools.supplemental_code import *
import tools.visualizer as visualizer


def read_data(pc_id=30, pc_exp=20):
    '''Extract all the principal components that can be used to represent face geometry'''

    bfm = h5py.File('../Data/model2017-1_face12_nomouth.h5', 'r')

    mean_id = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
    mean_id = np.reshape(mean_id, (-1, 3))
    PCAbasis_id = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    PCAbasis_id = np.reshape(PCAbasis_id, (28588, 3, 199))
    E_id = PCAbasis_id[:, :, :pc_id]
    PCAvar_id = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)
    sigma_id = np.sqrt(PCAvar_id[:pc_id])

    mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
    mean_exp = np.reshape(mean_exp, (-1, 3))
    PCAbasis_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    PCAbasis_exp = np.reshape(PCAbasis_exp, (28588, 3, 100))
    E_exp =  PCAbasis_exp[:, :, :pc_exp]
    PCAvar_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)
    sigma_exp = np.sqrt(PCAvar_exp[:pc_exp])

    triangle_topology = np.asarray(bfm['shape/representer/cells'], dtype=np.float32).T

    color_vertex = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color_vertex = np.reshape(color_vertex, (-1, 3))

    data = {'mean_id': mean_id, 'E_id': E_id, 'sigma_id': sigma_id, 'mean_exp': mean_exp, 'E_exp': E_exp, 'sigma_exp': sigma_exp, 'color_vertex': color_vertex, 'triangles': triangle_topology}

    for key, var in data.items():
        if key != 'color_vertex' and key != 'triangles':
            data[key] = torch.from_numpy(var).double()

    return data


def sample_latent_params():
    '''alpha and delta can be estimated for a given image of a face, or randomly sampled like here'''
    alpha = nn.Parameter(Variable(torch.tensor(np.random.uniform(-1, 1, 30)), requires_grad=True))
    delta = nn.Parameter(Variable(torch.tensor(np.random.uniform(-1, 1, 20)), requires_grad=True))
    return alpha, delta


def represent_geometry(alpha, delta, data=None, draw_pointcloud=False, save_model=False, save_dir=None, file_name=None):
    '''Represent facial geometry as a point cloud using Blanz & Vetter\'s  multilinear PCA model G(alpha, delta) (equation (1) in assignment).'''
    mean_id = data['mean_id']
    E_id = data['E_id']
    sigma_id = data['sigma_id']
    mean_exp = data['mean_exp']
    E_exp = data['E_exp']
    sigma_exp = data['sigma_exp']
    vertex_color = data['color_vertex']
    triangles = data['triangles']

    identity = mean_id + torch.matmul(E_id, alpha * sigma_id)
    expression =  mean_exp + torch.matmul(E_exp, delta * sigma_exp)
    G = identity + expression

    if draw_pointcloud:
        visualizer.draw_pointcloud(G.detach().numpy(), colors=vertex_color)

    if save_model:
        save_obj(save_dir + file_name, G.detach().numpy(), vertex_color, triangles)

    return G


if __name__ == "__main__":
    save_dir_name = '../Results/3Dmodels/'
    data = read_data()

    # Section 2: generate different faces for different alpha and delta values
    for sample in range(3):
        file_name = 's2_' + '__face{}_.obj'.format(sample)
        alpha, delta = sample_latent_params()
        face_vertices = represent_geometry(alpha, delta, data, draw_pointcloud=True, save_model=True, save_dir=save_dir_name, file_name=file_name)