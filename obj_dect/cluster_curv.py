import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def curv_pt(pcd):
    """
    Takes tensor Pointcloud, pcd, of low curvature points and performs DBSCAN to cluster close points together.
    Outputs clusters as differently colored points with corresponding bounding boxes. 
    
    Param:
    pcd : open3d.t.geometry.PointCloud
        Pointcloud tensor of low curvature points (output of get_curv)

    Return:
    None
    """
    pcd_pt = np.asarray(pcd.points)

    labels = np.array(pcd.cluster_dbscan(eps=0.2, min_points=10, print_progress=True))
    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    clust_pts = []
    for clust in range(max_label+1):
        clust_list = []
        clust_pts.append(clust_list)

    if len(set(labels)) != 1:
        for i, lab in enumerate(labels):
            clust_pts[lab].append(pcd_pt[i])

        vis_list = []
        for clust in range(max_label+1):
            mine = o3d.geometry.PointCloud()
            mine.points = o3d.utility.Vector3dVector(np.array(clust_pts[clust]))
            try: 
                bbox = mine.get_minimal_oriented_bounding_box()
                bbox.color = (1,0,0)
                vis_list.append(bbox)
            except: 
                print("Not enough points to construct simplex")

        mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size = 1.0)
        mesh.translate(pcd.get_center())
        vis_list.append(pcd)
        vis_list.append(mesh)
        o3d.visualization.draw_geometries(vis_list)
    else:
        print("No Clusters found...")