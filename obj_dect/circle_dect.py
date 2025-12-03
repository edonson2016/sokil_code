import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from rotate import rotate

def vert_slice(pcd):
    pcd = pcd.to_legacy()
    pcd = rotate(pcd)
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0)
    pcd_pt = np.asarray(pcd.points)
    y_min = min(pcd_pt[:,1])
    y_max = max(pcd_pt[:,1])
    print(y_min)
    print(y_max)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.1,  
        ransac_n=10,
        num_iterations=1000
    )
    pcd_plane = pcd.select_by_index(inliers, invert = False)
    y_mean = np.mean(np.asarray(pcd_plane.points)[:,1])
    print(y_mean)
    inc = (y_max - y_mean)/50
    print(inc)
    curr_y = y_mean
    while curr_y < y_max:
        pcd_pt_filt = list(filter(lambda x: x[1] > curr_y - 0.1 and x[1] < curr_y + 0.1, pcd_pt))
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(pcd_pt_filt)
        o3d.visualization.draw_geometries([temp_pcd, mesh])
        curr_y += inc