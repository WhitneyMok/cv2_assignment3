import numpy as np
import open3d as o3d

# from readPcd import read_mat_data


def draw_pointclouds(source, target):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)

    o3d.visualization.draw_geometries([pcd_source, pcd_target])


def draw_pointcloud(source):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    o3d.visualization.draw_geometries([pcd_source])