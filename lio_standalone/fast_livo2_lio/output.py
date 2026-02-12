"""Trajectory and point cloud output writers.

Supports TUM format and CSV format for odometry and per-scan point clouds.
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [qx, qy, qz, qw].

    Uses scipy which returns [x, y, z, w] order, matching the C++ TUM output
    format (Eigen's q.x(), q.y(), q.z(), q.w()).
    """
    r = Rotation.from_matrix(R)
    return r.as_quat()  # [x, y, z, w]


def write_tum(filepath: str, trajectory: list):
    """Write trajectory in TUM format.

    Args:
        filepath: Output file path.
        trajectory: List of (timestamp, pos(3,), quat(4,)) tuples.
                   Quaternion in [qx, qy, qz, qw] order.
    """
    with open(filepath, 'w') as f:
        for ts, pos, q in trajectory:
            f.write(f"{ts:.6f} {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f} "
                    f"{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}\n")


def write_odometry_csv(filepath: str, trajectory: list):
    """Write trajectory as CSV with header.

    Columns: timestamp,tx,ty,tz,qx,qy,qz,qw

    Args:
        filepath: Output file path.
        trajectory: List of (timestamp, pos(3,), quat(4,)) tuples.
    """
    with open(filepath, 'w') as f:
        f.write("timestamp,tx,ty,tz,qx,qy,qz,qw\n")
        for ts, pos, q in trajectory:
            f.write(f"{ts:.6f},{pos[0]:.6f},{pos[1]:.6f},{pos[2]:.6f},"
                    f"{q[0]:.6f},{q[1]:.6f},{q[2]:.6f},{q[3]:.6f}\n")


def write_scan_csv(filepath: str, points: np.ndarray,
                   intensities: np.ndarray = None):
    """Write a single point cloud scan as CSV.

    Columns: x,y,z[,intensity]

    Args:
        filepath: Output file path.
        points: (N, 3) array of xyz coordinates.
        intensities: Optional (N,) array of intensity values.
    """
    with open(filepath, 'w') as f:
        if intensities is not None and len(intensities) == len(points):
            f.write("x,y,z,intensity\n")
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f},{points[i, 1]:.6f},"
                        f"{points[i, 2]:.6f},{intensities[i]:.1f}\n")
        else:
            f.write("x,y,z\n")
            for i in range(len(points)):
                f.write(f"{points[i, 0]:.6f},{points[i, 1]:.6f},"
                        f"{points[i, 2]:.6f}\n")
