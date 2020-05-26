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
from supplemental_code import *
import cv2
from PIL import Image, ImageFilter


import visualizer



def getRotationMatrix(omega):
    '''
    omega should be an array where the elements correspond to the angle along x, y, or z axis
    '''
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(omega[0]), -math.sin(omega[0])],
                    [0, math.sin(omega[0]), math.cos(omega[0])]
                    ])

    R_y = np.array([[math.cos(omega[1]), 0, math.sin(omega[1])],
                    [0, 1, 0],
                    [-math.sin(omega[1]), 0, math.cos(omega[1])]
                    ])

    R_z = np.array([[math.cos(omega[2]), -math.sin(omega[2]), 0],
                    [math.sin(omega[2]), math.cos(omega[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    # R = np.dot(R_x, np.dot(R_y, R_z))

    return R


def get_transformationmatrix(omegas, t):
    R = getRotationMatrix(omegas)
    temp = np.hstack((R, t))
    bottom_row = np.matrix([[0, 0, 0, 1]])
    T = np.vstack((temp, bottom_row))
    return T


def get_ViewportMatrix(v_left=0, v_right=1600, v_top=0, v_bottom=1200):
    S = np.asarray([[(v_right-v_left)/2, 0, 0, 0],
                    [0, (v_top-v_bottom)/2, 0, 0],
                    [0, 0, 0.5, 0],
                    [0, 0, 0, 1]]) # TODO v_top v_bottom which ones

    T = np.asarray([[1, 0, 0, (v_left + v_right)/2],
                    [0, 1, 0, (v_bottom + v_top)/2],
                    [0, 0, 0.5, 0.5],
                    [0, 0, 0, 1]])
    V = np.dot(T, S)
    return V

def get_PerspectiveProjectionMatrix(FOV=90, n=0.1, f=100):
    aspect_ratio = 1600 / 1200
    t = math.tan((FOV/2) * (math.pi/180)) * n
    # t = math.tan(FOV/2) * n
    # b = -t
    r = t * aspect_ratio
    b = -t * aspect_ratio
    l = b

    P = np.asarray([[(2*n)/(r-l), 0, 0, 0],
                    [0, (2*n)/(t-b), 0, 0],
                    [(r+l)/(r-l), (t+b)/(t-b), -(f+n)/(f-n), -1],
                    [0, 0, -(2*f*n)/(f-n), 0]])

    return P

def get_pi():
    V = get_ViewportMatrix()
    P = get_PerspectiveProjectionMatrix()

    pi = np.dot(V, P)
    return pi

def cartesian2homogeneous(cartesian_points, axis=1, yo=False):
    if axis == 1:
        homogenous_points = np.hstack((cartesian_points, np.ones((cartesian_points.shape[0], 1))))
    elif axis == 0:
        homogenous_points = np.vstack((cartesian_points, np.ones((1, cartesian_points.shape[0]))))

    elif yo:
        homogenous_points = np.vstack((cartesian_points, np.ones((1, cartesian_points.shape[0]))))
    #
    # ax = 0 if cartesian_points.shape[0] > cartesian_points.shape[1] else 1
    # if ax == 0:
    #     homogenous_points = np.vstack((cartesian_points,  np.ones((1, cartesian_points.shape[0]))))
    # elif ax == 1:
    #     homogenous_points = np.hstack((cartesian_points, np.ones((cartesian_points.shape[0], 1))))
    return homogenous_points

def homogenous2cartesian(homogenous_points, axis=0):
    if axis == 0:
        return homogenous_points[:3, :].T
    elif axis == 1:
        return homogenous_points[:, :3]


def alt_homogenous2cartesian(transformed_face_4d):
    d_hat = transformed_face_4d[-1, :]
    img = transformed_face_4d[:3, :] / d_hat
    return img

def degrees2radians(rotations_in_degrees):
    rotation_in_radians = []
    for rotation in rotations_in_degrees:
        rotation_in_radians.append(rotation * (math.pi / 180) )
    rotation_in_radians_np = np.asarray(rotation_in_radians)
    return rotation_in_radians_np


# def rigid_transformation(face, omegas, xyz_translations):
def rigid_transformation(face_vertices_3d_homog_coord, omegas, xyz_translations):
    # face_vertices_3d_homog_coord = np.vstack((face.T, np.ones((1, face.shape[0]))))
    # face_vertices_3d_homog_coord = cartesian2homogeneous(face.T, axis=None, yo=True)

    T = get_transformationmatrix(omegas, xyz_translations)
    pi = get_pi()

    face_vertices_homog_coord = np.dot(pi, np.dot(T, face_vertices_3d_homog_coord.T))

    face_vertices_2d_cart_coord = face_vertices_homog_coord[:2, :]       # divide by 1, no need

    return face_vertices_homog_coord, face_vertices_2d_cart_coord


if __name__ == "__main__":

    draw_pointcloud = False
    save_3dmodel = False
    save_dir = './3Dmodels/'
    nr = random.randint(0, 1000)
    mean_id, E_id, sigma_id, mean_exp, E_exp, sigma_exp, color_vertex, triangles = read_data()

    file_name = str(nr) + '.obj'
    alpha, delta = get_latent_params()

    face_vertices = represent_geometry(alpha, delta, mean_id, E_id, sigma_id, mean_exp, E_exp, sigma_exp, color_vertex, triangles, draw_pointcloud=draw_pointcloud, save_model=save_3dmodel,
                                       vertex_color=color_vertex, file_name=file_name)
    face_vertices_homog = cartesian2homogeneous(face_vertices, axis=1)

    # Q.a
    save_model = False

    xyz_translations = np.matrix([[0], [0], [0]])

    omegas = degrees2radians([0, 10, 0])
    T1 = get_transformationmatrix(omegas, xyz_translations)
    face_only_rotated_ten_degrees_pos = np.dot(face_vertices_homog, T1)
    # face_only_rotated_ten_degrees_pos = np.dot(T1, face_vertices_homog)
    face_only_rotated_ten_degrees_pos = homogenous2cartesian(face_only_rotated_ten_degrees_pos, axis=1)

    omegas = degrees2radians([0, -10, 0])
    # omegas = [0, -10, 0]
    T2 = get_transformationmatrix(omegas, xyz_translations)
    face_only_rotated_ten_degrees_neg = np.dot(face_vertices_homog, T2)
    # face_only_rotated_ten_degrees_neg = np.dot(T2, face_vertices_homog)
    face_only_rotated_ten_degrees_neg = homogenous2cartesian(face_only_rotated_ten_degrees_neg, axis=1)

    # omegas = [0, 10, 0]
    # xyz_translations = np.matrix([[0], [0], [-500]])
    # T33 = get_transformationmatrix(omegas, xyz_translations)
    # face_only_rotated_testtest = np.dot(face_vertices_homog, T33)
    # # face_only_rotated_ten_degrees_neg = np.dot(T2, face_vertices_homog)
    # face_only_rotated_testest = homogenous2cartesian(face_only_rotated_testtest, axis=1)

    if save_model:
        file_name = 's3_Qa_' + str(nr) + '__ORIGINAL_.obj'
        save_obj(save_dir + file_name, face_vertices, color_vertex, triangles)
        # file_name = 's3_Qa_' + str(nr) + '__pos_10y_rot_.obj'
        # save_obj(save_dir + file_name, face_only_rotated_ten_degrees_pos, color_vertex, triangles)
        # file_name =  's3_Qa_' + str(nr) + '__neg_10y_rot_.obj'
        # save_obj(save_dir + file_name, face_only_rotated_ten_degrees_neg, color_vertex, triangles)
        # file_name = 's3_Qa_' + str(nr) + '__sdfsdfAL_.obj'
        # save_obj(save_dir + file_name, face_only_rotated_testest, color_vertex, triangles)
    print()



    # Q.b
    save_model= False

    f = open("./Data/Landmarks68_model2017-1_face12_nomouth.anl", "r")
    vertex_indcs_annot = [int(i.strip('\n')) for i in f.readlines()]

    xyz_translations = np.matrix([[0], [0], [-500]])
    # omegas = [0, 10, 0]
    omegas = degrees2radians([0, 10, 0])
    transformed_face_4d, transformed_face_2ddd = rigid_transformation(face_vertices_homog, omegas, xyz_translations) #TODO:  better var name
    # transformed_face_3d = homogenous2cartesian(transformed_face_4d, axis=0)
    transformed_face_3d = alt_homogenous2cartesian(transformed_face_4d).T
    transformed_face_2d = transformed_face_3d[:, :2]

    if save_model:
        # file_name = 's3_Qb_' + str(nr) + '__dsfdsfdsfdsfdsfdsz_trans.obj'
        # save_obj(save_dir + file_name, img, color_vertex, triangles)
        file_name = 's3_Qb_' + str(nr) + '__pos_10y_rot__-500z_trans.obj'
        # save_obj(save_dir + file_name, transformed_face_3d, color_vertex, triangles)
        save_obj(save_dir + file_name, transformed_face_3d, color_vertex, triangles)

    landmarks = transformed_face_2d[vertex_indcs_annot, :]
    x = [ele[0] for ele in landmarks[:, 0].tolist()]
    y = [ele[0] for ele in landmarks[:, 1].tolist()]
    plt.scatter(x, y)
    plt.show()





