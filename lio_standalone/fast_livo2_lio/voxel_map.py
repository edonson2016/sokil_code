"""Voxel map with octree subdivision, PCA plane fitting, and iterative EKF
state estimation.

Matches C++ VoxelOctoTree, VoxelPlane, and VoxelMapManager classes from
src/voxel_map.cpp.
"""
import math
import numpy as np
from .so3 import skew, skew_batch, exp_so3_dt
from .state import StatesGroup, DIM_STATE
from .types import PointWithVar, PointToPlane
from .numba_kernels import (
    calc_body_cov_batch_jit,
    compute_residual_math,
    build_jacobian_hth_jit,
    batch_transform_covariance_jit,
    batch_transform_points_jit,
    build_residual_list_jit,
)
from numba import types
from numba.typed import Dict as NumbaDict


DEG2RAD = math.pi / 180.0


def calc_body_cov(pb: np.ndarray, range_inc: float, degree_inc: float) -> np.ndarray:
    """Compute measurement covariance in LiDAR body frame.

    Matches C++ calcBodyCov (voxel_map.cpp:15-34).

    Args:
        pb: (3,) point in body frame.
        range_inc: Range error standard deviation (meters).
        degree_inc: Angular error standard deviation (degrees).

    Returns:
        (3, 3) covariance matrix.
    """
    p = pb.copy()
    if p[2] == 0:
        p[2] = 0.0001

    r = np.linalg.norm(p)
    range_var = range_inc * range_inc
    sin_deg = math.sin(degree_inc * DEG2RAD)
    direction_var = np.array([[sin_deg ** 2, 0.0],
                              [0.0, sin_deg ** 2]])

    direction = p / r
    direction_hat = skew(direction)

    # Build orthonormal tangent basis
    if abs(direction[2]) > 1e-6:
        base1 = np.array([1.0, 1.0, -(direction[0] + direction[1]) / direction[2]])
    elif abs(direction[1]) > 1e-6:
        base1 = np.array([1.0, -(direction[0] + direction[2]) / direction[1], 1.0])
    else:
        base1 = np.array([-(direction[1] + direction[2]) / direction[0], 1.0, 1.0])
    base1 = base1 / np.linalg.norm(base1)
    base2 = np.cross(base1, direction)
    base2 = base2 / np.linalg.norm(base2)

    N = np.column_stack([base1, base2])  # (3, 2)
    A = r * direction_hat @ N  # (3, 2)

    cov = (np.outer(direction, direction) * range_var +
           A @ direction_var @ A.T)
    return cov


def calc_body_cov_batch(pb_array: np.ndarray, range_inc: float,
                        degree_inc: float) -> np.ndarray:
    """Vectorized body covariance for N points at once.

    Equivalent to calling calc_body_cov() N times but ~50-100x faster.

    Args:
        pb_array: (N, 3) points in body frame.
        range_inc: Range error standard deviation (meters).
        degree_inc: Angular error standard deviation (degrees).

    Returns:
        (N, 3, 3) array of covariance matrices.
    """
    N = pb_array.shape[0]
    p = pb_array.copy()
    # Avoid division by zero on z
    zero_z = p[:, 2] == 0
    p[zero_z, 2] = 0.0001

    r = np.linalg.norm(p, axis=1)  # (N,)
    range_var = range_inc * range_inc
    sin_deg = math.sin(degree_inc * DEG2RAD)
    dir_var_scalar = sin_deg * sin_deg

    direction = p / r[:, np.newaxis]  # (N, 3)

    # Build orthonormal tangent basis (vectorized with masks)
    base1 = np.zeros((N, 3))
    abs_dir = np.abs(direction)

    # Case 1: |z| > 1e-6
    m1 = abs_dir[:, 2] > 1e-6
    # Case 2: not m1 and |y| > 1e-6
    m2 = (~m1) & (abs_dir[:, 1] > 1e-6)
    # Case 3: everything else
    m3 = (~m1) & (~m2)

    if np.any(m1):
        base1[m1, 0] = 1.0
        base1[m1, 1] = 1.0
        base1[m1, 2] = -(direction[m1, 0] + direction[m1, 1]) / direction[m1, 2]
    if np.any(m2):
        base1[m2, 0] = 1.0
        base1[m2, 1] = -(direction[m2, 0] + direction[m2, 2]) / direction[m2, 1]
        base1[m2, 2] = 1.0
    if np.any(m3):
        base1[m3, 0] = -(direction[m3, 1] + direction[m3, 2]) / direction[m3, 0]
        base1[m3, 1] = 1.0
        base1[m3, 2] = 1.0

    base1_norm = np.linalg.norm(base1, axis=1, keepdims=True)
    base1 = base1 / base1_norm

    base2 = np.cross(base1, direction)
    base2_norm = np.linalg.norm(base2, axis=1, keepdims=True)
    base2 = base2 / base2_norm

    # N matrix: (N, 3, 2) = stack of [base1, base2] columns
    N_mat = np.stack([base1, base2], axis=2)  # (N, 3, 2)

    # direction_hat = skew(direction): (N, 3, 3)
    direction_hat = skew_batch(direction)

    # A = r * direction_hat @ N: (N, 3, 2)
    A = np.einsum('nij,njk->nik', direction_hat, N_mat) * r[:, np.newaxis, np.newaxis]

    # cov = outer(direction, direction) * range_var + A @ dir_var @ A.T
    # outer part: (N, 3, 3)
    cov = np.einsum('ni,nj->nij', direction, direction) * range_var

    # A @ diag(dir_var, dir_var) @ A.T = dir_var_scalar * (A @ A.T)
    # A is (N, 3, 2), A @ A.T is (N, 3, 3)
    cov += dir_var_scalar * np.einsum('nik,njk->nij', A, A)

    return cov


class VoxelPlane:
    """Fitted plane within a voxel node.

    Matches C++ VoxelPlane struct from include/voxel_map.h.
    """
    __slots__ = ['center', 'normal', 'y_normal', 'x_normal',
                 'covariance', 'plane_var', 'min_eigen', 'mid_eigen',
                 'max_eigen', 'radius', 'd', 'points_size',
                 'is_plane', 'is_init', 'is_update']

    def __init__(self):
        self.center = np.zeros(3)
        self.normal = np.zeros(3)
        self.y_normal = np.zeros(3)
        self.x_normal = np.zeros(3)
        self.covariance = np.zeros((3, 3))
        self.plane_var = np.zeros((6, 6))
        self.min_eigen = 1.0
        self.mid_eigen = 1.0
        self.max_eigen = 1.0
        self.radius = 0.0
        self.d = 0.0
        self.points_size = 0
        self.is_plane = False
        self.is_init = False
        self.is_update = False


class VoxelOctoTree:
    """Octree node within a voxel. Supports hierarchical plane fitting.

    Matches C++ VoxelOctoTree class from src/voxel_map.cpp.
    """
    __slots__ = ['plane_ptr', 'leaves', 'layer', 'max_layer',
                 'points_size_threshold', 'max_points_num',
                 'planer_threshold', 'voxel_center', 'quarter_length',
                 'temp_points', 'new_points', 'init_octo',
                 'update_enable', 'octo_state', 'layer_init_num',
                 'update_size_threshold']

    def __init__(self, max_layer: int, layer: int,
                 points_size_threshold: int, max_points_num: int,
                 planer_threshold: float):
        self.plane_ptr = VoxelPlane()
        self.leaves = [None] * 8
        self.layer = layer
        self.max_layer = max_layer
        self.points_size_threshold = points_size_threshold
        self.max_points_num = max_points_num
        self.planer_threshold = planer_threshold
        self.voxel_center = np.zeros(3)
        self.quarter_length = 0.0
        self.temp_points = []   # list of (point_w(3,), var(3,3))
        self.new_points = 0
        self.init_octo = False
        self.update_enable = True
        self.octo_state = 0  # 0=leaf, 1=branch
        self.layer_init_num = []
        self.update_size_threshold = 5

    def init_plane(self, points):
        """Fit plane via PCA eigendecomposition.

        Matches C++ VoxelOctoTree::init_plane (voxel_map.cpp:55-135).

        Args:
            points: list of (point_w(3,), var(3,3)) tuples.
        """
        plane = self.plane_ptr
        plane.plane_var = np.zeros((6, 6))
        plane.covariance = np.zeros((3, 3))
        plane.center = np.zeros(3)
        plane.normal = np.zeros(3)
        plane.points_size = len(points)
        plane.radius = 0.0
        n = plane.points_size

        if n == 0:
            plane.is_plane = False
            return

        # Compute covariance
        sum_pw = np.zeros(3)
        sum_pw2 = np.zeros((3, 3))
        for pw, var in points:
            sum_pw += pw
            sum_pw2 += np.outer(pw, pw)

        plane.center = sum_pw / n
        plane.covariance = sum_pw2 / n - np.outer(plane.center, plane.center)

        # Eigendecomposition (general solver to match C++ EigenSolver)
        evals_raw, evecs_raw = np.linalg.eig(plane.covariance)
        evals = evals_raw.real
        evecs = evecs_raw.real

        # Find min/max/mid indices
        evals_min_idx = np.argmin(evals)
        evals_max_idx = np.argmax(evals)
        evals_mid_idx = 3 - evals_min_idx - evals_max_idx

        evec_min = evecs[:, evals_min_idx]
        evec_mid = evecs[:, evals_mid_idx]
        evec_max = evecs[:, evals_max_idx]

        J_Q = np.eye(3) / n

        if evals[evals_min_idx] < self.planer_threshold:
            # Compute plane covariance via Jacobian propagation
            for pw, var in points:
                J = np.zeros((6, 3))
                F = np.zeros((3, 3))
                for m in range(3):
                    if m != evals_min_idx:
                        denom = n * (evals[evals_min_idx] - evals[m])
                        if abs(denom) < 1e-12:
                            F[m, :] = 0.0
                        else:
                            # C++: (1,3) = (1,3) * scalar * (3,3)
                            # (pw - center) is (3,), treat as row vector
                            diff = (pw - plane.center).reshape(1, 3)
                            outer_sum = (np.outer(evecs[:, m], evecs[:, evals_min_idx]) +
                                         np.outer(evecs[:, evals_min_idx], evecs[:, m]))
                            F_m = (diff / denom) @ outer_sum  # (1,3)
                            F[m, :] = F_m.ravel()
                    else:
                        F[m, :] = 0.0
                J[0:3, :] = evecs @ F
                J[3:6, :] = J_Q
                plane.plane_var += J @ var @ J.T

            plane.normal = evec_min.copy()
            plane.y_normal = evec_mid.copy()
            plane.x_normal = evec_max.copy()
            plane.min_eigen = evals[evals_min_idx]
            plane.mid_eigen = evals[evals_mid_idx]
            plane.max_eigen = evals[evals_max_idx]
            plane.radius = math.sqrt(abs(evals[evals_max_idx]))
            plane.d = -(plane.normal @ plane.center)
            plane.is_plane = True
            plane.is_update = True
            if not plane.is_init:
                plane.is_init = True
        else:
            plane.is_update = True
            plane.is_plane = False

    def init_octo_tree(self):
        """Initialize octree node: fit plane or subdivide.

        Matches C++ VoxelOctoTree::init_octo_tree (voxel_map.cpp:137-161).
        """
        if len(self.temp_points) > self.points_size_threshold:
            self.init_plane(self.temp_points)
            if self.plane_ptr.is_plane:
                self.octo_state = 0
                if len(self.temp_points) > self.max_points_num:
                    self.update_enable = False
                    self.temp_points = []
                    self.new_points = 0
            else:
                self.octo_state = 1
                self._cut_octo_tree()
            self.init_octo = True
            self.new_points = 0

    def _cut_octo_tree(self):
        """Subdivide into 8 children and recursively init.

        Matches C++ VoxelOctoTree::cut_octo_tree (voxel_map.cpp:163-217).
        """
        if self.layer >= self.max_layer:
            self.octo_state = 0
            return

        for pw, var in self.temp_points:
            xyz = [0, 0, 0]
            if pw[0] > self.voxel_center[0]:
                xyz[0] = 1
            if pw[1] > self.voxel_center[1]:
                xyz[1] = 1
            if pw[2] > self.voxel_center[2]:
                xyz[2] = 1
            leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]

            if self.leaves[leafnum] is None:
                child_thresh = (self.layer_init_num[self.layer + 1]
                                if self.layer + 1 < len(self.layer_init_num)
                                else self.points_size_threshold)
                child = VoxelOctoTree(
                    self.max_layer, self.layer + 1,
                    child_thresh, self.max_points_num,
                    self.planer_threshold
                )
                child.layer_init_num = self.layer_init_num
                child.voxel_center[0] = self.voxel_center[0] + (2 * xyz[0] - 1) * self.quarter_length
                child.voxel_center[1] = self.voxel_center[1] + (2 * xyz[1] - 1) * self.quarter_length
                child.voxel_center[2] = self.voxel_center[2] + (2 * xyz[2] - 1) * self.quarter_length
                child.quarter_length = self.quarter_length / 2
                self.leaves[leafnum] = child

            self.leaves[leafnum].temp_points.append((pw, var))
            self.leaves[leafnum].new_points += 1

        for i in range(8):
            child = self.leaves[i]
            if child is not None:
                if len(child.temp_points) > child.points_size_threshold:
                    child.init_plane(child.temp_points)
                    if child.plane_ptr.is_plane:
                        child.octo_state = 0
                        if len(child.temp_points) > child.max_points_num:
                            child.update_enable = False
                            child.temp_points = []
                            child.new_points = 0
                    else:
                        child.octo_state = 1
                        child._cut_octo_tree()
                    child.init_octo = True
                    child.new_points = 0

    def update_octo_tree(self, pw: np.ndarray, var: np.ndarray):
        """Insert a new point into the octree.

        Matches C++ VoxelOctoTree::UpdateOctoTree (voxel_map.cpp:219-290).
        """
        pv = (pw, var)
        if not self.init_octo:
            self.new_points += 1
            self.temp_points.append(pv)
            if len(self.temp_points) > self.points_size_threshold:
                self.init_octo_tree()
        else:
            if self.plane_ptr.is_plane:
                if self.update_enable:
                    self.new_points += 1
                    self.temp_points.append(pv)
                    if self.new_points > self.update_size_threshold:
                        self.init_plane(self.temp_points)
                        self.new_points = 0
                    if len(self.temp_points) >= self.max_points_num:
                        self.update_enable = False
                        self.temp_points = []
                        self.new_points = 0
            else:
                if self.layer < self.max_layer:
                    xyz = [0, 0, 0]
                    if pw[0] > self.voxel_center[0]:
                        xyz[0] = 1
                    if pw[1] > self.voxel_center[1]:
                        xyz[1] = 1
                    if pw[2] > self.voxel_center[2]:
                        xyz[2] = 1
                    leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]

                    if self.leaves[leafnum] is not None:
                        self.leaves[leafnum].update_octo_tree(pw, var)
                    else:
                        child_thresh = (self.layer_init_num[self.layer + 1]
                                        if self.layer + 1 < len(self.layer_init_num)
                                        else self.points_size_threshold)
                        child = VoxelOctoTree(
                            self.max_layer, self.layer + 1,
                            child_thresh, self.max_points_num,
                            self.planer_threshold
                        )
                        child.layer_init_num = self.layer_init_num
                        child.voxel_center[0] = self.voxel_center[0] + (2 * xyz[0] - 1) * self.quarter_length
                        child.voxel_center[1] = self.voxel_center[1] + (2 * xyz[1] - 1) * self.quarter_length
                        child.voxel_center[2] = self.voxel_center[2] + (2 * xyz[2] - 1) * self.quarter_length
                        child.quarter_length = self.quarter_length / 2
                        self.leaves[leafnum] = child
                        child.update_octo_tree(pw, var)
                else:
                    if self.update_enable:
                        self.new_points += 1
                        self.temp_points.append(pv)
                        if self.new_points > self.update_size_threshold:
                            self.init_plane(self.temp_points)
                            self.new_points = 0
                        if len(self.temp_points) > self.max_points_num:
                            self.update_enable = False
                            self.temp_points = []
                            self.new_points = 0

    def find_correspond(self, pw: np.ndarray):
        """Find the leaf octree node containing a query point.

        Matches C++ VoxelOctoTree::find_correspond (voxel_map.cpp:292-305).
        """
        if not self.init_octo or self.plane_ptr.is_plane or self.layer >= self.max_layer:
            return self

        xyz = [0, 0, 0]
        xyz[0] = 1 if pw[0] > self.voxel_center[0] else 0
        xyz[1] = 1 if pw[1] > self.voxel_center[1] else 0
        xyz[2] = 1 if pw[2] > self.voxel_center[2] else 0
        leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]

        if self.leaves[leafnum] is not None:
            return self.leaves[leafnum].find_correspond(pw)
        return self


def _voxel_key(point_w: np.ndarray, voxel_size: float):
    """Compute voxel hash key from world point.

    Matches C++ voxel location computation (voxel_map.cpp:561-567).
    """
    loc = point_w / voxel_size
    # Floor toward negative infinity (matching C++ int64_t cast with -1 offset)
    loc_int = np.empty(3, dtype=np.int64)
    for j in range(3):
        v = loc[j]
        if v < 0:
            v -= 1.0
        loc_int[j] = int(v)
    return (loc_int[0], loc_int[1], loc_int[2])


class VoxelMapManager:
    """Top-level voxel map: hash table of voxel locations to octrees,
    plus state estimation via iterative EKF.

    Matches C++ VoxelMapManager class from src/voxel_map.cpp.
    """

    def __init__(self, voxel_size: float = 0.5, max_layer: int = 2,
                 max_iterations: int = 5,
                 planer_threshold: float = 0.01,
                 sigma_num: float = 3.0,
                 dept_err: float = 0.02, beam_err: float = 0.05,
                 max_points_num: int = 50,
                 layer_init_num: list = None,
                 ext_R: np.ndarray = None, ext_T: np.ndarray = None):
        self.voxel_map = {}  # dict[(int,int,int)] -> VoxelOctoTree
        self.voxel_size = voxel_size
        self.max_layer = max_layer
        self.max_iterations = max_iterations
        self.planer_threshold = planer_threshold
        self.sigma_num = sigma_num
        self.dept_err = dept_err
        self.beam_err = beam_err
        self.max_points_num = max_points_num
        self.layer_init_num = layer_init_num or [5, 5, 5, 5, 5]

        self.ext_R = ext_R if ext_R is not None else np.eye(3)
        self.ext_T = ext_T if ext_T is not None else np.zeros(3)

        # Plane cache for JIT residual builder
        self._plane_cache_valid = False
        self._key_type = types.UniTuple(types.int64, 3)

    def _collect_planes_from_octo(self, octo, planes_list):
        """Recursively collect all planes from an octree into a flat list."""
        pp = octo.plane_ptr
        if pp.is_plane:
            planes_list.append((pp.normal, pp.d, pp.center, pp.plane_var, pp.radius))
            return
        # Recurse into children
        if octo.layer < self.max_layer:
            for child in octo.leaves:
                if child is not None:
                    self._collect_planes_from_octo(child, planes_list)

    def _rebuild_plane_cache(self):
        """Rebuild the flat plane cache from the current voxel map.

        This extracts all plane data from octrees into contiguous arrays
        so the JIT residual builder can access them without Python overhead.
        """
        voxel_map = self.voxel_map
        n_voxels = len(voxel_map)

        # Collect all planes
        all_planes = []  # list of (normal, d, center, plane_var, radius)
        voxel_starts = {}  # key -> start index
        voxel_counts = {}  # key -> count

        for key, octo in voxel_map.items():
            planes_list = []
            self._collect_planes_from_octo(octo, planes_list)
            if planes_list:
                voxel_starts[key] = len(all_planes)
                voxel_counts[key] = len(planes_list)
                all_planes.extend(planes_list)

        n_planes = len(all_planes)
        if n_planes == 0:
            self._plane_cache_valid = False
            return

        # Build flat arrays
        plane_normals = np.empty((n_planes, 3))
        plane_d = np.empty(n_planes)
        plane_centers = np.empty((n_planes, 3))
        plane_vars = np.empty((n_planes, 6, 6))
        plane_radii = np.empty(n_planes)

        for i, (normal, d, center, pvar, radius) in enumerate(all_planes):
            plane_normals[i] = normal
            plane_d[i] = d
            plane_centers[i] = center
            plane_vars[i] = pvar
            plane_radii[i] = radius

        # Build Numba typed dicts
        starts_dict = NumbaDict.empty(self._key_type, types.int64)
        counts_dict = NumbaDict.empty(self._key_type, types.int64)
        vcmap_dict = NumbaDict.empty(self._key_type, types.int64)

        voxel_centers_arr = np.empty((n_voxels, 3))
        voxel_ql_arr = np.empty(n_voxels)

        vi = 0
        for key, octo in voxel_map.items():
            nb_key = (np.int64(key[0]), np.int64(key[1]), np.int64(key[2]))
            if key in voxel_starts:
                starts_dict[nb_key] = np.int64(voxel_starts[key])
                counts_dict[nb_key] = np.int64(voxel_counts[key])
            vcmap_dict[nb_key] = np.int64(vi)
            voxel_centers_arr[vi] = octo.voxel_center
            voxel_ql_arr[vi] = octo.quarter_length
            vi += 1

        self._pc_normals = plane_normals
        self._pc_d = plane_d
        self._pc_centers = plane_centers
        self._pc_vars = plane_vars
        self._pc_radii = plane_radii
        self._pc_starts = starts_dict
        self._pc_counts = counts_dict
        self._pc_vcmap = vcmap_dict
        self._pc_voxel_centers = voxel_centers_arr[:vi]
        self._pc_voxel_ql = voxel_ql_arr[:vi]
        self._plane_cache_valid = True

    def build_voxel_map(self, points_body: np.ndarray,
                        points_world: np.ndarray,
                        state: StatesGroup):
        """Build the initial voxel map from the first frame.

        Matches C++ VoxelMapManager::BuildVoxelMap (voxel_map.cpp:532-591).
        """
        rot_ext = state.rot_end @ self.ext_R
        rot_cov = state.cov[0:3, 0:3]
        t_cov = state.cov[3:6, 3:6]

        # Batch covariance computation (JIT parallel)
        n_pts = len(points_body)
        body_cov_all = calc_body_cov_batch_jit(points_body, self.dept_err, self.beam_err)  # (N,3,3)
        p_imu_all = (self.ext_R @ points_body.T).T + self.ext_T  # (N,3)

        var_all = batch_transform_covariance_jit(
            body_cov_all, p_imu_all, rot_ext, rot_cov, t_cov)  # (N,3,3)

        # Batch voxel key computation
        loc = points_world / self.voxel_size
        loc_neg = loc < 0
        loc_adj = loc.copy()
        loc_adj[loc_neg] -= 1.0
        loc_int = loc_adj.astype(np.int64)

        for i in range(n_pts):
            pw = points_world[i]
            var = var_all[i]
            key = (loc_int[i, 0], loc_int[i, 1], loc_int[i, 2])

            if key in self.voxel_map:
                self.voxel_map[key].temp_points.append((pw.copy(), var))
                self.voxel_map[key].new_points += 1
            else:
                octo = VoxelOctoTree(
                    self.max_layer, 0, self.layer_init_num[0],
                    self.max_points_num, self.planer_threshold
                )
                octo.layer_init_num = self.layer_init_num
                octo.quarter_length = self.voxel_size / 4
                octo.voxel_center[0] = (0.5 + key[0]) * self.voxel_size
                octo.voxel_center[1] = (0.5 + key[1]) * self.voxel_size
                octo.voxel_center[2] = (0.5 + key[2]) * self.voxel_size
                octo.temp_points.append((pw.copy(), var))
                octo.new_points += 1
                self.voxel_map[key] = octo

        for octo in self.voxel_map.values():
            octo.init_octo_tree()

        self._rebuild_plane_cache()

    def update_voxel_map(self, pv_list: list):
        """Incrementally update the voxel map with new points.

        Matches C++ VoxelMapManager::UpdateVoxelMap (voxel_map.cpp:609-641).

        Args:
            pv_list: list of (point_w(3,), var(3,3)) tuples.
        """
        for pw, var in pv_list:
            key = _voxel_key(pw, self.voxel_size)
            if key in self.voxel_map:
                self.voxel_map[key].update_octo_tree(pw, var)
            else:
                octo = VoxelOctoTree(
                    self.max_layer, 0, self.layer_init_num[0],
                    self.max_points_num, self.planer_threshold
                )
                octo.layer_init_num = self.layer_init_num
                octo.quarter_length = self.voxel_size / 4
                octo.voxel_center[0] = (0.5 + key[0]) * self.voxel_size
                octo.voxel_center[1] = (0.5 + key[1]) * self.voxel_size
                octo.voxel_center[2] = (0.5 + key[2]) * self.voxel_size
                self.voxel_map[key] = octo
                octo.update_octo_tree(pw, var)

        self._rebuild_plane_cache()

    def _build_single_residual(self, pv_point_w, pv_point_b, pv_var,
                                pv_body_var, current_octo, current_layer,
                                is_success, prob, result):
        """Traverse octree to find best matching plane for a point.

        Matches C++ build_single_residual (voxel_map.cpp:713-786).
        Uses JIT-compiled compute_residual_math for the heavy math.
        """
        plane_ptr = current_octo.plane_ptr
        if plane_ptr.is_plane:
            success, this_prob, dis_signed = compute_residual_math(
                pv_point_w, plane_ptr.normal, plane_ptr.d,
                plane_ptr.center, plane_ptr.plane_var,
                pv_var, self.sigma_num, plane_ptr.radius)

            if success and this_prob > prob[0]:
                is_success[0] = True
                prob[0] = this_prob
                result['point_b'] = pv_point_b
                result['point_w'] = pv_point_w
                result['normal'] = plane_ptr.normal
                result['center'] = plane_ptr.center
                result['plane_var'] = plane_ptr.plane_var
                result['body_cov'] = pv_body_var
                result['d'] = plane_ptr.d
                result['layer'] = current_layer
                result['dis_to_plane'] = dis_signed
            return

        # Not a plane: recurse into children
        if current_layer < self.max_layer:
            leaves = current_octo.leaves
            for leafnum in range(8):
                child = leaves[leafnum]
                if child is not None:
                    self._build_single_residual(
                        pv_point_w, pv_point_b, pv_var, pv_body_var,
                        child, current_layer + 1,
                        is_success, prob, result
                    )

    def build_residual_list(self, pv_list):
        """Find plane correspondences for all points.

        Matches C++ BuildResidualListOMP (voxel_map.cpp:643-711).
        Single-threaded Python version.

        Args:
            pv_list: list of dicts with keys 'point_b', 'point_w', 'var', 'body_var'

        Returns:
            List of PointToPlane measurements.
        """
        ptpl_list = []

        for pv in pv_list:
            pw = pv['point_w']
            key = _voxel_key(pw, self.voxel_size)

            if key not in self.voxel_map:
                continue

            current_octo = self.voxel_map[key]
            is_success = [False]
            prob = [0.0]
            result = {}

            self._build_single_residual(
                pw, pv['point_b'], pv['var'], pv['body_var'],
                current_octo, 0, is_success, prob, result
            )

            if not is_success[0]:
                # Try neighbor voxel
                loc = pw / self.voxel_size
                near_key = list(key)
                center = current_octo.voxel_center
                ql = current_octo.quarter_length

                if loc[0] > center[0] + ql:
                    near_key[0] += 1
                elif loc[0] < center[0] - ql:
                    near_key[0] -= 1
                if loc[1] > center[1] + ql:
                    near_key[1] += 1
                elif loc[1] < center[1] - ql:
                    near_key[1] -= 1
                if loc[2] > center[2] + ql:
                    near_key[2] += 1
                elif loc[2] < center[2] - ql:
                    near_key[2] -= 1

                near_key = tuple(near_key)
                if near_key in self.voxel_map:
                    self._build_single_residual(
                        pw, pv['point_b'], pv['var'], pv['body_var'],
                        self.voxel_map[near_key], 0, is_success, prob, result
                    )

            if is_success[0]:
                ptpl = PointToPlane(
                    point_b=result['point_b'].copy(),
                    point_w=result['point_w'].copy(),
                    normal=result['normal'].copy(),
                    center=result['center'].copy(),
                    plane_var=result['plane_var'].copy(),
                    body_cov=result['body_cov'].copy(),
                    d=result['d'],
                    layer=result['layer'],
                    dis_to_plane=result['dis_to_plane'],
                )
                ptpl_list.append(ptpl)

        return ptpl_list

    def build_residual_list_batch(self, points_body, points_world,
                                   cov_all, body_cov_all):
        """Find plane correspondences using JIT-compiled plane cache.

        Uses pre-flattened plane data and Numba typed Dict for the entire
        loop to run in compiled code, eliminating Python interpreter overhead.

        Args:
            points_body: (N, 3) body frame points
            points_world: (N, 3) world frame points
            cov_all: (N, 3, 3) point covariances
            body_cov_all: (N, 3, 3) body covariances

        Returns:
            List of PointToPlane measurements.
        """
        if not self._plane_cache_valid:
            return None  # Returns None to indicate empty, checked below

        (out_pb, out_pw, out_normal, out_center, out_pvar,
         out_bcov, out_d, out_dis, out_count) = build_residual_list_jit(
            points_body, points_world, cov_all, body_cov_all,
            self.voxel_size, self.sigma_num,
            self._pc_normals, self._pc_d, self._pc_centers,
            self._pc_vars, self._pc_radii,
            self._pc_starts, self._pc_counts,
            self._pc_vcmap, self._pc_voxel_centers, self._pc_voxel_ql)

        M = out_count
        if M == 0:
            return None

        # Return sliced arrays directly (no PointToPlane creation)
        return (out_pb[:M], out_pw[:M], out_normal[:M], out_center[:M],
                out_pvar[:M], out_bcov[:M], out_d[:M], out_dis[:M])

    def state_estimation(self, state: StatesGroup,
                         state_propagat: StatesGroup,
                         points_body: np.ndarray) -> StatesGroup:
        """Iterative EKF state estimation with point-to-plane residuals.

        Matches C++ VoxelMapManager::StateEstimation (voxel_map.cpp:338-511).

        Args:
            state: Current state estimate (modified in place).
            state_propagat: Propagated state (prior).
            points_body: (N, 3) downsampled points in LiDAR body frame.

        Returns:
            Updated state.
        """
        n_pts = len(points_body)

        # Precompute body covariances (JIT parallel)
        body_cov_all = calc_body_cov_batch_jit(points_body, self.dept_err, self.beam_err)  # (N,3,3)

        rematch_num = 0
        I_STATE = np.eye(DIM_STATE)
        G = np.zeros((DIM_STATE, DIM_STATE))
        H_T_H = np.zeros((DIM_STATE, DIM_STATE))

        for iter_count in range(self.max_iterations):
            # Transform points to world frame with current state (JIT parallel)
            rot_var = state.cov[0:3, 0:3]
            t_var = state.cov[3:6, 3:6]

            pw_all, p_imu_all = batch_transform_points_jit(
                self.ext_R, self.ext_T, state.rot_end, state.pos_end, points_body)

            # Batch covariance (JIT parallel)
            cov_all = batch_transform_covariance_jit(
                body_cov_all, p_imu_all, state.rot_end, rot_var, t_var)

            # Build residual list (JIT-compiled with plane cache)
            result = self.build_residual_list_batch(
                points_body, pw_all, cov_all, body_cov_all)

            if result is None:
                print(f"[ LIO ] No effective features at iteration {iter_count}")
                break

            (ptpl_point_b, ptpl_point_w, ptpl_normals, ptpl_centers,
             ptpl_plane_vars, ptpl_body_covs, ptpl_d_arr, ptpl_dis) = result
            effct_feat_num = len(ptpl_point_b)

            if effct_feat_num < 1:
                print(f"[ LIO ] No effective features at iteration {iter_count}")
                break

            # Build Jacobian H, H^T R^{-1} H, H^T R^{-1} z  all in one JIT pass
            rot_end_T = np.ascontiguousarray(state.rot_end.T)
            H_T_H_66, HTz = build_jacobian_hth_jit(
                ptpl_point_b, ptpl_normals, ptpl_centers,
                ptpl_plane_vars, ptpl_body_covs, ptpl_dis,
                self.ext_R, self.ext_T,
                state_propagat.rot_end, state_propagat.pos_end,
                state.rot_end, rot_end_T)
            H_T_H[:] = 0.0
            H_T_H[0:6, 0:6] = H_T_H_66

            try:
                state_cov_inv = np.linalg.inv(state.cov)
            except np.linalg.LinAlgError:
                state_cov_inv = np.linalg.pinv(state.cov)

            K_1 = np.linalg.inv(H_T_H + state_cov_inv)

            G[:] = 0.0
            G[:, 0:6] = K_1[:, 0:6] @ H_T_H_66

            vec = state_propagat.boxminus(state)
            solution = K_1[:, 0:6] @ HTz + vec - G[:, 0:6] @ vec[0:6]

            state.boxplus_inplace(solution)

            # Convergence check
            rot_add = solution[0:3]
            t_add = solution[3:6]
            flg_converged = (np.linalg.norm(rot_add) * 57.3 < 0.01 and
                             np.linalg.norm(t_add) * 100 < 0.015)

            # Rematch logic
            if flg_converged or (rematch_num == 0 and iter_count == self.max_iterations - 2):
                rematch_num += 1

            if rematch_num >= 2 or iter_count == self.max_iterations - 1:
                state.cov = (I_STATE - G) @ state.cov
                break

        return state
