import numpy as np
import open3d as o3d

''''These functions are for drawing pointclouds'''

def draw_pointclouds(source, target):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)

    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(target)

    o3d.visualization.draw_geometries([pcd_source, pcd_target])


def draw_pointcloud(source, colors=None):
    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(source)
    if colors is not None:
        pcd_source.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd_source])