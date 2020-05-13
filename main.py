import sys
import os
import h5py
import numpy as np
import random
# from supplemental_code import *
import visualizer


def read_data(pc_id=30, pc_exp=20):
    bfm = h5py.File('./Data/model2017-1_bfm_nomouth.h5', 'r')

    mean_id = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
    mean_id = np.reshape(mean_id, (-1, 3))
    PCAbasis_id = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    PCAbasis_id = np.reshape(PCAbasis_id, (53149, 3, 199))
    E_id = PCAbasis_id[:, :, :30]
    PCAvar_id = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)
    sigma_id = np.sqrt(PCAvar_id[:30])

    mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
    mean_exp = np.reshape(mean_exp, (-1, 3))
    PCAbasis_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    PCAbasis_exp = np.reshape(PCAbasis_id, (53149, 3, 199))
    E_exp =  PCAbasis_exp[:, :, :30]
    PCAvar_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)
    sigma_exp = np.sqrt(PCAvar_exp[:30])

    triangle_topology = np.asarray(bfm['shape/representer/cells'], dtype=np.float32)
    triangle_topology = np.reshape(triangle_topology, (-1, 3))

    color_vertex = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    color_vertex = np.reshape(color_vertex, (-1, 3))
    return mean_id, E_id, sigma_id, mean_exp, E_exp, sigma_exp, color_vertex

def get_latent_params():
    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    return alpha, delta

def represent_geometry(alpha, delta, draw_pointcloud=False, save_model=False, vertex_color=None, file_name=None):

    G = mean_id + np.matmul(E_id, np.multiply(alpha, sigma_id)) + mean_exp + np.matmul(E_id, np.multiply(alpha, sigma_id))

    if draw_pointcloud:
        visualizer.draw_pointcloud(G)

    if save_model:
        save_obj(save_dir + file_name, G, vertex_color, )


    return G

draw_pointcloud = True
save_3dmodel = True
save_dir = '/3Dmodels/'
nr = random.randint(0,100)
mean_id, E_id, sigma_id, mean_exp, E_exp, sigma_exp, color_vertex = read_data()

file_name = str(nr)
alpha, delta = get_latent_params()

represent_geometry(alpha, delta, draw_pointcloud=draw_pointcloud, save_model=save_3dmodel, vertex_color=color_vertex, file_name=file_name)
print()