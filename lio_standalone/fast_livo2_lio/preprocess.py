"""Point cloud preprocessing: blind zone filter, point decimation, time sort.

Matches the non-feature-extraction path in C++ Preprocess class (src/preprocess.cpp).
All operations are vectorized with NumPy.
"""
import numpy as np


class Preprocessor:
    """Preprocess raw LiDAR scans: blind zone filter and point decimation."""

    def __init__(self, blind: float = 0.8, point_filter_num: int = 1):
        """
        Args:
            blind: Minimum range in meters. Points closer are removed.
            point_filter_num: Keep every Nth point (1 = keep all).
        """
        self.blind_sqr = blind * blind
        self.point_filter_num = max(1, point_filter_num)

    def process(self, points_xyz: np.ndarray, intensities: np.ndarray,
                time_offsets_ms: np.ndarray):
        """Apply blind zone filter and point decimation.

        Args:
            points_xyz: (N, 3) point coordinates in body frame.
            intensities: (N,) intensity values.
            time_offsets_ms: (N,) per-point time offset in ms from scan start.

        Returns:
            Tuple of (filtered_xyz, filtered_intensities, filtered_times_ms),
            sorted by time offset.
        """
        if len(points_xyz) == 0:
            return points_xyz, intensities, time_offsets_ms

        # Compute squared range
        r2 = np.sum(points_xyz ** 2, axis=1)

        # Point decimation mask: keep every Nth point
        decimate_mask = np.zeros(len(points_xyz), dtype=bool)
        decimate_mask[::self.point_filter_num] = True

        # Blind zone filter: remove near points
        valid_mask = (r2 >= self.blind_sqr) & decimate_mask

        # Remove NaN/Inf points
        finite_mask = np.all(np.isfinite(points_xyz), axis=1)
        valid_mask = valid_mask & finite_mask

        xyz = points_xyz[valid_mask]
        inten = intensities[valid_mask]
        times = time_offsets_ms[valid_mask]

        # Sort by time offset
        sort_idx = np.argsort(times)
        return xyz[sort_idx], inten[sort_idx], times[sort_idx]
