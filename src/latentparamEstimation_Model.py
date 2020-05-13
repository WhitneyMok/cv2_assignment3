import h5py
import dlib
import math
import numpy as np
from .morphable_model import *
from .pinhole_cam_model import *
from tools.supplemental_code import *
import tools.visualizer as visualizer
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self, data, alpha, delta, omega, t):
        super(Model, self).__init__()
        self.data = data
        self.alpha = alpha
        self.delta = delta
        self.omega = omega
        self.t = t

    def forward(self):
        face_vertices = represent_geometry(self.alpha, self.delta, self.data)
        face_vertices_homog = cartesian2homogeneous(face_vertices)
        rigidly_transformed_face = rigid_transformation(face_vertices_homog, self.omega, self.t)
        face_2Dprojection = torch.transpose(get_u_v_projection(rigidly_transformed_face), 0, 1)
        landmarks = get_bfm_landmarks(face_2Dprojection)
        return landmarks
