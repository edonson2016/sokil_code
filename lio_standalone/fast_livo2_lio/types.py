"""Data structures used throughout the LIO pipeline.

Matches C++ types from include/common_lib.h and include/utils/types.h.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Pose6D:
    """IMU pose at a specific time offset within a scan.

    Matches C++ Pose6D from include/utils/types.h.
    """
    offset_time: float = 0.0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyr: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rot: np.ndarray = field(default_factory=lambda: np.eye(3))


@dataclass
class ImuData:
    """Single IMU measurement."""
    timestamp: float = 0.0
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyr: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class LidarScan:
    """Preprocessed LiDAR scan with per-point timestamps."""
    header_time: float = 0.0
    points: np.ndarray = None       # (N, 3) xyz in body frame
    intensities: np.ndarray = None  # (N,)
    timestamps_ms: np.ndarray = None  # (N,) offset from header_time in ms


@dataclass
class PointWithVar:
    """Point with body/world coordinates and covariance.

    Matches C++ pointWithVar from include/common_lib.h:102-123.
    """
    point_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_i: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    var: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    body_var: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    point_crossmat: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class PointToPlane:
    """Point-to-plane measurement residual.

    Matches C++ PointToPlane from include/voxel_map.h.
    """
    point_b: np.ndarray = field(default_factory=lambda: np.zeros(3))
    point_w: np.ndarray = field(default_factory=lambda: np.zeros(3))
    normal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    plane_var: np.ndarray = field(default_factory=lambda: np.zeros((6, 6)))
    body_cov: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    d: float = 0.0
    layer: int = 0
    dis_to_plane: float = 0.0
