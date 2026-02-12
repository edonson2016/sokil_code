"""Voxel grid downsampling using NumPy.

Replaces pcl::VoxelGrid<PointType> from the C++ pipeline.
"""
import numpy as np


def voxel_grid_downsample(points: np.ndarray, leaf_size: float) -> np.ndarray:
    """Downsample a point cloud using voxel grid filtering.

    For each occupied voxel, the centroid of all points within it is returned.

    Args:
        points: (N, 3) point coordinates.
        leaf_size: Voxel edge length in meters.

    Returns:
        (M, 3) downsampled point coordinates (centroids).
    """
    if len(points) == 0:
        return points.copy()

    # Quantize to voxel indices
    voxel_idx = np.floor(points / leaf_size).astype(np.int64)

    # Encode 3D index to a single hashable key per point
    # Shift to non-negative for stable hashing
    mins = voxel_idx.min(axis=0)
    shifted = voxel_idx - mins
    dims = shifted.max(axis=0) + 1

    # Linear index
    linear = (shifted[:, 0] * dims[1] * dims[2] +
              shifted[:, 1] * dims[2] +
              shifted[:, 2])

    # Find unique voxels and compute centroids
    unique_keys, inverse = np.unique(linear, return_inverse=True)
    n_voxels = len(unique_keys)

    centroids = np.zeros((n_voxels, 3))
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)

    for dim in range(3):
        centroids[:, dim] = np.bincount(
            inverse, weights=points[:, dim], minlength=n_voxels
        ) / counts

    return centroids
