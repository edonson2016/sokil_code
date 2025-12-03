import numpy as np
from scipy.spatial.transform import Rotation as R

def rotate(pcd):
    """
    Uses RANSAC to detect plane within tensor Pointcloud, pcd. Takes normal of plane and constructs rotation matrix
    such that the detected plane in the rotated pcd will have a normal of (0,1,0). 
    
    Param:
    pcd : open3d.t.geometry.PointCloud
        Unrotated Pointcloud tensor of environment

    Return:
    pcd : open3d.t.geometry.PointCloud
        Rotated Pointcloud tensor of environment
    """

    plane_model, _ = pcd.segment_plane(
        distance_threshold=0.1,  
        ransac_n=10,
        num_iterations=1000
    )

    target_norm = np.array([0,1,0])
    current_norm = plane_model[:-1] / np.linalg.norm(plane_model[:-1])

    pcd_r = R.align_vectors(target_norm, current_norm)[0].as_matrix()
    pcd.rotate(pcd_r)

    return pcd
