"""SO(3) operations: Rodrigues formula, skew-symmetric matrix, Log map.

Matches the C++ implementations in include/utils/so3_math.h exactly.
"""
import numpy as np


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector.

    Args:
        v: (3,) vector

    Returns:
        (3, 3) skew-symmetric matrix such that skew(v) @ w == cross(v, w)
    """
    return np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0]
    ])


def exp_so3(ang: np.ndarray) -> np.ndarray:
    """Exponential map: so(3) -> SO(3) via Rodrigues formula.

    Matches C++ Exp(v1, v2, v3) with threshold 1e-7.

    Args:
        ang: (3,) angle-axis vector (rotation axis * angle in radians)

    Returns:
        (3, 3) rotation matrix
    """
    ang_norm = np.linalg.norm(ang)
    if ang_norm > 1e-7:
        r_axis = ang / ang_norm
        K = skew(r_axis)
        return np.eye(3) + np.sin(ang_norm) * K + (1.0 - np.cos(ang_norm)) * (K @ K)
    else:
        return np.eye(3)


def exp_so3_dt(ang_vel: np.ndarray, dt: float) -> np.ndarray:
    """Exponential map with angular velocity and time delta.

    Matches C++ Exp(ang_vel, dt).

    Args:
        ang_vel: (3,) angular velocity in rad/s
        dt: time delta in seconds

    Returns:
        (3, 3) rotation matrix
    """
    ang_vel_norm = np.linalg.norm(ang_vel)
    if ang_vel_norm > 1e-7:
        r_axis = ang_vel / ang_vel_norm
        K = skew(r_axis)
        r_ang = ang_vel_norm * dt
        return np.eye(3) + np.sin(r_ang) * K + (1.0 - np.cos(r_ang)) * (K @ K)
    else:
        return np.eye(3)


def log_so3(R: np.ndarray) -> np.ndarray:
    """Logarithm map: SO(3) -> so(3).

    Matches C++ Log(R) exactly: clamp trace at 3-1e-6, small-angle fallback.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (3,) angle-axis vector
    """
    trace = np.trace(R)
    theta = 0.0 if trace > 3.0 - 1e-6 else np.arccos(np.clip(0.5 * (trace - 1.0), -1.0, 1.0))
    K = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if abs(theta) < 0.001:
        return 0.5 * K
    else:
        return 0.5 * theta / np.sin(theta) * K


def skew_batch(v_array: np.ndarray) -> np.ndarray:
    """Batch skew-symmetric matrices from (N,3) vectors.

    Args:
        v_array: (N, 3) array of vectors.

    Returns:
        (N, 3, 3) array of skew-symmetric matrices.
    """
    N = v_array.shape[0]
    result = np.zeros((N, 3, 3))
    result[:, 0, 1] = -v_array[:, 2]
    result[:, 0, 2] = v_array[:, 1]
    result[:, 1, 0] = v_array[:, 2]
    result[:, 1, 2] = -v_array[:, 0]
    result[:, 2, 0] = -v_array[:, 1]
    result[:, 2, 1] = v_array[:, 0]
    return result


def exp_so3_batch(ang_array: np.ndarray) -> np.ndarray:
    """Batch exponential map: (N,3) angle-axis vectors -> (N,3,3) rotation matrices.

    Uses Rodrigues formula with vectorized operations.

    Args:
        ang_array: (N, 3) angle-axis vectors.

    Returns:
        (N, 3, 3) rotation matrices.
    """
    N = ang_array.shape[0]
    norms = np.linalg.norm(ang_array, axis=1)  # (N,)
    result = np.tile(np.eye(3), (N, 1, 1))  # (N, 3, 3) identity

    # Only compute for non-trivial rotations
    mask = norms > 1e-7
    if not np.any(mask):
        return result

    axes = ang_array[mask] / norms[mask, np.newaxis]  # (M, 3)
    K = skew_batch(axes)  # (M, 3, 3)
    K2 = np.einsum('nij,njk->nik', K, K)  # (M, 3, 3) = K @ K

    sin_t = np.sin(norms[mask])[:, np.newaxis, np.newaxis]  # (M, 1, 1)
    cos_t = (1.0 - np.cos(norms[mask]))[:, np.newaxis, np.newaxis]  # (M, 1, 1)

    result[mask] = result[mask] + sin_t * K + cos_t * K2
    return result


def rot_to_euler(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to Euler angles (XYZ convention).

    Matches C++ RotMtoEuler.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        (3,) Euler angles [roll, pitch, yaw] in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return np.array([x, y, z])
