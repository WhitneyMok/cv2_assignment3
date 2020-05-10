import sys
import os
import h5py
import numpy as np
import visualizer


def read_data(pc_id=30, pc_exp=20):
    bfm = h5py.File('./Data/model2017-1_bfm_nomouth.h5', 'r')

    mean_id = np.asarray(bfm['shape/model/mean'], dtype=np.float32)
    mean_id = np.reshape(mean_id, (-1, 3))
    PCAbasis_id = np.asarray(bfm['shape/model/pcaBasis'], dtype=np.float32)
    PCAbasis_id = np.reshape(PCAbasis_id, (53149, 3, 199))
    E_id = PCAbasis_id[:, :, :30]
    PCAvar_id = np.asarray(bfm['shape/model/pcaVariance'], dtype=np.float32)
    sigma_id = PCAvar_id[:30]

    mean_exp = np.asarray(bfm['expression/model/mean'], dtype=np.float32)
    mean_exp = np.reshape(mean_exp, (-1, 3))
    PCAbasis_exp = np.asarray(bfm['expression/model/pcaBasis'], dtype=np.float32)
    PCAbasis_exp = np.reshape(PCAbasis_id, (53149, 3, 199))
    E_exp =  PCAbasis_exp[:, :, :30]
    PCAvar_exp = np.asarray(bfm['expression/model/pcaVariance'], dtype=np.float32)
    sigma_exp = PCAvar_exp[:30]

    triangle_topology = np.asarray(bfm['shape/representer/cells'], dtype=np.float32)

    color_vertex = np.asarray(bfm['color/model/mean'], dtype=np.float32)
    return mean_id, E_id, sigma_id, mean_exp, E_exp, sigma_exp

def get_latent_params():
    alpha = np.random.uniform(-1, 1, 30)
    delta = np.random.uniform(-1, 1, 20)
    return alpha, delta

def represent_geometry(alpha, delta, draw_pointcloud=True):
    
    G = mean_id + (E_id * np.dot(alpha, sigma_id)) + mean_exp + (E_exp * np.dot(delta, sigma_exp))

    # if draw_pointcloud:
    #     visualizer.drawpointcloud(..)
    return G



mean_id, E_id, sigma_id, mean_exp, E_exp, sigma_exp = read_data()

alpha, delta = get_latent_params()

represent_geometry(alpha, delta)
print()