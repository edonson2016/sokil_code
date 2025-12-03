import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def get_curv(pcd, low_prop = 0.1):
    """
    Takes pcd and performs K-nearest neighbor search of each point to produce clusters of the 30 close points.
    Then performs PCA to extract eigenvalues which are used to claculate curvature.
    The points in the pcd are sorted by curvature and the "low_prop" lowest are chosen and returned as a filtered point cloud

    Param:
    pcd : open3d.t.geometry.PointCloud
        Pointcloud tensor of environment chunk

    low_prop : double (0 to 1)
        Proportion of low eigenvalue points to select and return

    Return:
    open3d.geometry.PointCloud
        Pointcloud tensor of low eigenvalue points in environment chunk
    """

    pcd_pt = pcd.point["positions"].numpy()
    pcd = pcd.to_legacy()
    y_min = min(pcd_pt[:,1])
    y_max = max(pcd_pt[:,1])
    print(len(pcd_pt))
    pcd_pt_filt = list(filter(lambda x: x[1] > y_min and x[1] < y_min + (0.2*(y_max-y_min)), pcd_pt))
    print(len(pcd_pt_filt))
    pcd.points = o3d.utility.Vector3dVector(pcd_pt_filt)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = []
    for i in range(len(pcd.points)):
        if i % 1000 == 0:
            print(i/len(pcd.points))
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], 30)
        neighbors = np.asarray(pcd.points)[idx, :]
        cov = np.cov(neighbors.T)
        eigvals, _ = np.linalg.eigh(cov)
        eigvals = np.sort(eigvals)
        curvature = eigvals[0] / eigvals.sum()
        curvatures.append((curvature,i))

    high_curv = list(sorted(curvatures, key = lambda x: x[0])[:int(len(pcd.points)*low_prop)])
    print(high_curv)

    curv_pcd = o3d.geometry.PointCloud()
    curv_pcd_pts = pcd_pt[[x[1] for x in high_curv],:]

    curv_pcd.points = o3d.utility.Vector3dVector(curv_pcd_pts)

    return curv_pcd

