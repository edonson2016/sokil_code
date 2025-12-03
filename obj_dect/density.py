import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def heat_map(pcd, y_cutoff, radius):
    """
    Takes :param pcd: and determines the point cloud density in a radius X radius square. This heatmap is displayed along with
    a heatmap representing the change in intensity from each radius X radius square to the other.
    
    Param:
    pcd : open3d.t.geometry.PointCloud
        Pointcloud tensor of environment

    y_cutoff : Double 
        Determines the highest y-value that will be considered when calculating mean_density

    Return:
    None
    """

    points = pcd.point["positions"].numpy()
    y_min = np.min(points[:, 1], axis=0)
    points[:, 1] -= y_min
    print(f"Before: {points.shape}")
    points = points[y_cutoff > points[:, 1]]
    print(f"After: {points.shape}")
    pcd = pcd.to_legacy()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    densities = []

    for point in points:
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, radius)
        densities.append(k)

    densities = np.array(densities)
    # Color points by density
    normalized_densities = (densities - densities.min()) / (densities.max() - densities.min())
    colors = plt.cm.jet(normalized_densities)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Visualize
    #o3d.visualization.draw_geometries([pcd])
    points = np.hstack((points, densities.reshape(-1, 1)))

    densities = points[:, 3]
    unit_square = 0.05

    x_min, x_max = np.min(points[:, 0], axis=0), np.max(points[:, 0], axis=0)
    z_min, z_max = np.min(points[:, 2], axis=0), np.max(points[:, 2], axis=0)
    x_bins = int((x_max - x_min)/unit_square) + 1
    z_bins = int((z_max - z_min)/unit_square) + 1
    density_grid = np.zeros(shape=(x_bins, z_bins))
    count_grid = np.zeros(shape=(x_bins, z_bins))

    # Assign each point to a grid cell and accumulate densities
    x_indices = ((points[:,0] - x_min) / unit_square).astype(int)
    z_indices = ((points[:,2] - z_min) / unit_square).astype(int)

    # Clip indices to be within bounds
    x_indices = np.clip(x_indices, 0, x_bins - 1)
    z_indices = np.clip(z_indices, 0, z_bins - 1)

    for i, d in enumerate(densities):
        xi, zi = x_indices[i], z_indices[i]
        density_grid[xi, zi] += d
        count_grid[xi, zi] += 1

    mean_density_grid = np.divide(density_grid, count_grid, 
                               out=np.zeros_like(density_grid), 
                               where=count_grid != 0)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(mean_density_grid.T, origin='lower', cmap='hot', interpolation='nearest',
            extent=[x_min, x_max, z_min, z_max])
    plt.colorbar(label='Points per cell')
    plt.xlabel('X coordinate')
    plt.ylabel('Z coordinate')
    plt.title(f'Neighborhood Point Density Below y = {y_cutoff} (cell size: {unit_square})')
    plt.show()

    
    ideal_shape = mean_density_grid.shape
    dx, dz = np.gradient(mean_density_grid)
    adjusted_grad = np.sqrt(dx**2 + dz**2).reshape((ideal_shape[0], -1))

    plt.figure(figsize=(10, 8))
    plt.imshow(adjusted_grad.T, origin='lower', cmap='hot', interpolation='nearest',
            extent=[x_min, x_max, z_min, z_max])
    plt.colorbar(label='Points per cell')
    plt.xlabel('X coordinate')
    plt.ylabel('Z coordinate')
    plt.title(f'Change in Neighborhood Point Density Below y = {y_cutoff} (cell size: {unit_square})')
    plt.show()


# CODE USE TO TEST density.py W/O HAVING TO RUN run_alg.py

# filename = "landmine_1_wmine.pcd"
# pcd_raw = o3d.io.read_point_cloud(filename)
# pcd_raw_pt = np.asarray(pcd_raw.points)
# pcd = o3d.t.geometry.PointCloud()
# pcd.point["positions"] = o3d.core.Tensor(pcd_raw_pt, dtype=o3d.core.Dtype.Float64)
# heat_map(pcd)

# coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
#     size=1.0,  # Length of each axis
#     origin=[5.6, 0, 3.2]  # Position at (0, 5, 0)
# )

# o3d.visualization.draw_geometries([pcd.to_legacy(), coord_frame])