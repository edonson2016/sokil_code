"""Numba JIT-compiled kernels for performance-critical inner loops.

These functions replace the Python-level math with machine-code compiled
equivalents via Numba's @njit decorator. Key targets:

1. calc_body_cov — single-point and parallel-batch body covariance
2. build_single_residual_math — plane distance / sigma / probability
3. build_jacobian_hth — batch Jacobian + H^T H + H^T z in one pass
4. undistort_bracket — per-bracket point transform
5. SO(3) operations — skew, exp_so3 (Rodrigues)
"""
import math
import numpy as np
from numba import njit, prange


# ─────────────────────────────────────────────────────────────
#  SO(3) primitives
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def skew_jit(v):
    """Skew-symmetric matrix from 3-vector."""
    result = np.zeros((3, 3))
    result[0, 1] = -v[2]
    result[0, 2] = v[1]
    result[1, 0] = v[2]
    result[1, 2] = -v[0]
    result[2, 0] = -v[1]
    result[2, 1] = v[0]
    return result


@njit(cache=True)
def exp_so3_jit(ang):
    """Exponential map so(3)->SO(3) via Rodrigues."""
    ang_norm = 0.0
    for j in range(3):
        ang_norm += ang[j] * ang[j]
    ang_norm = math.sqrt(ang_norm)

    if ang_norm > 1e-7:
        r0 = ang[0] / ang_norm
        r1 = ang[1] / ang_norm
        r2 = ang[2] / ang_norm

        K = np.zeros((3, 3))
        K[0, 1] = -r2
        K[0, 2] = r1
        K[1, 0] = r2
        K[1, 2] = -r0
        K[2, 0] = -r1
        K[2, 1] = r0

        # K @ K
        K2 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += K[i, k] * K[k, j]
                K2[i, j] = s

        sin_t = math.sin(ang_norm)
        cos_t = 1.0 - math.cos(ang_norm)

        result = np.eye(3)
        for i in range(3):
            for j in range(3):
                result[i, j] += sin_t * K[i, j] + cos_t * K2[i, j]
        return result
    else:
        return np.eye(3)


@njit(cache=True)
def exp_so3_dt_jit(ang_vel, dt):
    """Exponential map with angular velocity and time delta."""
    ang = np.empty(3)
    ang[0] = ang_vel[0] * dt
    ang[1] = ang_vel[1] * dt
    ang[2] = ang_vel[2] * dt
    return exp_so3_jit(ang)


# ─────────────────────────────────────────────────────────────
#  Matrix operations (3x3)
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def mat3_mul(A, B):
    """3x3 matrix multiply."""
    C = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C


@njit(cache=True)
def mat3_vec3_mul(A, v):
    """3x3 @ 3-vector."""
    r = np.empty(3)
    for i in range(3):
        s = 0.0
        for j in range(3):
            s += A[i, j] * v[j]
        r[i] = s
    return r


@njit(cache=True)
def mat3_transpose(A):
    """Transpose of 3x3."""
    T = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            T[i, j] = A[j, i]
    return T


@njit(cache=True)
def vec3_dot(a, b):
    """Dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(cache=True)
def vec3_norm(a):
    """Norm of 3-vector."""
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


@njit(cache=True)
def vec3_cross(a, b):
    """Cross product of two 3-vectors."""
    r = np.empty(3)
    r[0] = a[1] * b[2] - a[2] * b[1]
    r[1] = a[2] * b[0] - a[0] * b[2]
    r[2] = a[0] * b[1] - a[1] * b[0]
    return r


# ─────────────────────────────────────────────────────────────
#  Body covariance
# ─────────────────────────────────────────────────────────────

DEG2RAD = math.pi / 180.0


@njit(cache=True)
def calc_body_cov_jit(pb, range_inc, degree_inc):
    """Compute measurement covariance in LiDAR body frame (single point).

    Matches C++ calcBodyCov.
    """
    p = np.empty(3)
    p[0] = pb[0]
    p[1] = pb[1]
    p[2] = pb[2]
    if p[2] == 0.0:
        p[2] = 0.0001

    r = vec3_norm(p)
    range_var = range_inc * range_inc
    sin_deg = math.sin(degree_inc * DEG2RAD)
    dir_var_scalar = sin_deg * sin_deg

    direction = np.empty(3)
    direction[0] = p[0] / r
    direction[1] = p[1] / r
    direction[2] = p[2] / r

    # Orthonormal tangent basis
    base1 = np.empty(3)
    abs_d0 = abs(direction[0])
    abs_d1 = abs(direction[1])
    abs_d2 = abs(direction[2])

    if abs_d2 > 1e-6:
        base1[0] = 1.0
        base1[1] = 1.0
        base1[2] = -(direction[0] + direction[1]) / direction[2]
    elif abs_d1 > 1e-6:
        base1[0] = 1.0
        base1[1] = -(direction[0] + direction[2]) / direction[1]
        base1[2] = 1.0
    else:
        base1[0] = -(direction[1] + direction[2]) / direction[0]
        base1[1] = 1.0
        base1[2] = 1.0

    b1_norm = vec3_norm(base1)
    base1[0] /= b1_norm
    base1[1] /= b1_norm
    base1[2] /= b1_norm

    base2 = vec3_cross(base1, direction)
    b2_norm = vec3_norm(base2)
    base2[0] /= b2_norm
    base2[1] /= b2_norm
    base2[2] /= b2_norm

    # N = [base1, base2] (3,2)
    # A = r * skew(direction) @ N  (3,2)
    dir_hat = skew_jit(direction)

    # A columns: A0 = r * dir_hat @ base1, A1 = r * dir_hat @ base2
    A0 = mat3_vec3_mul(dir_hat, base1)
    A1 = mat3_vec3_mul(dir_hat, base2)
    for j in range(3):
        A0[j] *= r
        A1[j] *= r

    # cov = outer(direction, direction) * range_var + dir_var_scalar * (A @ A.T)
    # A @ A.T = outer(A0,A0) + outer(A1,A1)
    cov = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            cov[i, j] = (direction[i] * direction[j] * range_var +
                         dir_var_scalar * (A0[i] * A0[j] + A1[i] * A1[j]))
    return cov


@njit(parallel=True, cache=True)
def calc_body_cov_batch_jit(pb_array, range_inc, degree_inc):
    """Parallel batch body covariance for N points."""
    N = pb_array.shape[0]
    result = np.empty((N, 3, 3))
    for i in prange(N):
        result[i] = calc_body_cov_jit(pb_array[i], range_inc, degree_inc)
    return result


# ─────────────────────────────────────────────────────────────
#  Single residual math (plane distance, sigma, probability)
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def compute_residual_math(p_w, p_normal, p_d, p_center, p_plane_var,
                          pv_var, sigma_num, plane_radius):
    """Compute residual for a single point against a plane.

    Returns (is_success, prob, J_nq_6, sigma_l, dis_to_plane_signed)
    """
    dis_to_plane_signed = (p_normal[0] * p_w[0] + p_normal[1] * p_w[1] +
                           p_normal[2] * p_w[2] + p_d)
    dis_to_plane = abs(dis_to_plane_signed)

    dx = p_center[0] - p_w[0]
    dy = p_center[1] - p_w[1]
    dz = p_center[2] - p_w[2]
    dis_to_center = dx * dx + dy * dy + dz * dz

    range_val = dis_to_center - dis_to_plane * dis_to_plane
    if range_val > 0:
        range_dis = math.sqrt(range_val)
    else:
        range_dis = 0.0

    if range_dis > 3.0 * plane_radius:
        return False, 0.0, 0.0

    # J_nq = [p_w - center, -normal] (6,)
    J_nq = np.empty(6)
    J_nq[0] = p_w[0] - p_center[0]
    J_nq[1] = p_w[1] - p_center[1]
    J_nq[2] = p_w[2] - p_center[2]
    J_nq[3] = -p_normal[0]
    J_nq[4] = -p_normal[1]
    J_nq[5] = -p_normal[2]

    # sigma_l = J_nq @ plane_var @ J_nq + normal @ pv_var @ normal
    sigma_l = 0.0
    for i in range(6):
        tmp = 0.0
        for j in range(6):
            tmp += p_plane_var[i, j] * J_nq[j]
        sigma_l += J_nq[i] * tmp

    # normal @ pv_var @ normal
    for i in range(3):
        tmp = 0.0
        for j in range(3):
            tmp += pv_var[i, j] * p_normal[j]
        sigma_l += p_normal[i] * tmp

    if sigma_l <= 0 or dis_to_plane >= sigma_num * math.sqrt(sigma_l):
        return False, 0.0, dis_to_plane_signed

    this_prob = (1.0 / math.sqrt(sigma_l) *
                 math.exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l))
    return True, this_prob, dis_to_plane_signed


# ─────────────────────────────────────────────────────────────
#  Jacobian H + H^T R^{-1} H + H^T R^{-1} z  in one JIT pass
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def build_jacobian_hth_jit(
    ptpl_point_b,    # (M,3) body points
    ptpl_normals,    # (M,3) plane normals
    ptpl_centers,    # (M,3) plane centers
    ptpl_plane_vars, # (M,6,6)
    ptpl_body_covs,  # (M,3,3)
    ptpl_dis,        # (M,) signed distance to plane
    ext_R,           # (3,3) extrinsic rotation
    ext_T,           # (3,) extrinsic translation
    rot_prop,        # (3,3) propagated rotation (state_propagat.rot_end)
    pos_prop,        # (3,) propagated position
    rot_end,         # (3,3) current state rotation
    rot_end_T,       # (3,3) transpose of rot_end
):
    """Build Jacobian H, compute H^T R^{-1} H (6x6) and H^T R^{-1} z (6,).

    All in a single JIT pass over M matched points.

    Returns:
        H_T_H_66: (6,6) = sum_i h_i h_i^T / R_i
        HTz: (6,) = sum_i h_i * meas_i / R_i
    """
    M = ptpl_point_b.shape[0]
    H_T_H_66 = np.zeros((6, 6))
    HTz = np.zeros(6)

    RE = mat3_mul(rot_prop, ext_R)  # (3,3)

    for m in range(M):
        # p_imu = ext_R @ point_b + ext_T
        p_imu = mat3_vec3_mul(ext_R, ptpl_point_b[m])
        for j in range(3):
            p_imu[j] += ext_T[j]

        p_cross = skew_jit(p_imu)  # (3,3)

        # p_world = rot_prop @ p_imu + pos_prop
        p_world = mat3_vec3_mul(rot_prop, p_imu)
        for j in range(3):
            p_world[j] += pos_prop[j]

        # J_nq
        J_nq = np.empty(6)
        normal = ptpl_normals[m]
        center = ptpl_centers[m]
        J_nq[0] = p_world[0] - center[0]
        J_nq[1] = p_world[1] - center[1]
        J_nq[2] = p_world[2] - center[2]
        J_nq[3] = -normal[0]
        J_nq[4] = -normal[1]
        J_nq[5] = -normal[2]

        # sigma_l = J_nq @ plane_var @ J_nq
        sigma_l = 0.0
        plane_var = ptpl_plane_vars[m]
        for i in range(6):
            tmp = 0.0
            for j in range(6):
                tmp += plane_var[i, j] * J_nq[j]
            sigma_l += J_nq[i] * tmp

        # var = RE @ body_cov @ RE.T
        body_cov = ptpl_body_covs[m]
        RE_T = mat3_transpose(RE)
        tmp33 = mat3_mul(RE, body_cov)
        var = mat3_mul(tmp33, RE_T)

        # nTvn = normal.T @ var @ normal
        nTvn = 0.0
        for i in range(3):
            tmp_val = 0.0
            for j in range(3):
                tmp_val += var[i, j] * normal[j]
            nTvn += normal[i] * tmp_val

        R_inv = 1.0 / (0.001 + sigma_l + nTvn)

        # H row: [cross_mat @ rot_end.T @ normal, normal]
        cross_rotT = mat3_mul(p_cross, rot_end_T)
        A_vec = mat3_vec3_mul(cross_rotT, normal)

        h = np.empty(6)
        h[0] = A_vec[0]
        h[1] = A_vec[1]
        h[2] = A_vec[2]
        h[3] = normal[0]
        h[4] = normal[1]
        h[5] = normal[2]

        # Accumulate H^T R^{-1} H and H^T R^{-1} z
        meas = -ptpl_dis[m]
        for i in range(6):
            h_ri = h[i] * R_inv
            HTz[i] += h_ri * meas
            for j in range(6):
                H_T_H_66[i, j] += h_ri * h[j]

    return H_T_H_66, HTz


# ─────────────────────────────────────────────────────────────
#  Undistortion: per-bracket point transform
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def undistort_bracket_jit(pts_in_bracket, dt_arr, R_head, gyr_head,
                          vel_head, pos_head, acc_head, pos_end,
                          Lid_rot_to_IMU, Lid_offset_to_IMU,
                          extR_Ri, exrR_extT):
    """Undistort all points in a single IMU bracket.

    Args:
        pts_in_bracket: (M,3) points in body frame
        dt_arr: (M,) time offsets from bracket head
        R_head, gyr_head, vel_head, pos_head, acc_head: bracket head pose
        pos_end: (3,) state position at scan end
        Lid_rot_to_IMU: (3,3) LiDAR to IMU rotation
        Lid_offset_to_IMU: (3,) LiDAR to IMU translation
        extR_Ri: (3,3) = Lid_rot_to_IMU.T @ rot_end.T
        exrR_extT: (3,) = Lid_rot_to_IMU.T @ Lid_offset_to_IMU

    Returns:
        (M,3) compensated points
    """
    M = pts_in_bracket.shape[0]
    result = np.empty((M, 3))

    for i in range(M):
        dt = dt_arr[i]

        # R_i = R_head @ Exp(gyr_head * dt)
        omega = np.empty(3)
        omega[0] = gyr_head[0] * dt
        omega[1] = gyr_head[1] * dt
        omega[2] = gyr_head[2] * dt
        Exp = exp_so3_jit(omega)
        R_i = mat3_mul(R_head, Exp)

        # T_ei = pos_head + vel_head*dt + 0.5*acc_head*dt^2 - pos_end
        dt2 = 0.5 * dt * dt
        T_ei = np.empty(3)
        T_ei[0] = pos_head[0] + vel_head[0] * dt + acc_head[0] * dt2 - pos_end[0]
        T_ei[1] = pos_head[1] + vel_head[1] * dt + acc_head[1] * dt2 - pos_end[1]
        T_ei[2] = pos_head[2] + vel_head[2] * dt + acc_head[2] * dt2 - pos_end[2]

        # P_imu = LidR @ P_i + LidT
        P_imu = mat3_vec3_mul(Lid_rot_to_IMU, pts_in_bracket[i])
        P_imu[0] += Lid_offset_to_IMU[0]
        P_imu[1] += Lid_offset_to_IMU[1]
        P_imu[2] += Lid_offset_to_IMU[2]

        # R_i @ P_imu + T_ei
        Ri_Pimu = mat3_vec3_mul(R_i, P_imu)
        Ri_Pimu[0] += T_ei[0]
        Ri_Pimu[1] += T_ei[1]
        Ri_Pimu[2] += T_ei[2]

        # extR_Ri @ Ri_Pimu - exrR_extT
        comp = mat3_vec3_mul(extR_Ri, Ri_Pimu)
        result[i, 0] = comp[0] - exrR_extT[0]
        result[i, 1] = comp[1] - exrR_extT[1]
        result[i, 2] = comp[2] - exrR_extT[2]

    return result


# ─────────────────────────────────────────────────────────────
#  Forward without IMU: batch undistortion
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def forward_without_imu_undistort_jit(points_xyz, dt_arr, bias_g,
                                       rot_end_T, vel_end):
    """Undistort points without IMU (constant velocity model).

    Args:
        points_xyz: (N,3) points
        dt_arr: (N,) time deltas (scan_end - point_time)
        bias_g: (3,) gyro bias
        rot_end_T: (3,3) transpose of rot_end
        vel_end: (3,) velocity at scan end

    Returns:
        (N,3) undistorted points
    """
    N = points_xyz.shape[0]
    result = np.empty((N, 3))

    # base_vel = -rot_end.T @ vel_end
    base_vel = mat3_vec3_mul(rot_end_T, vel_end)
    base_vel[0] = -base_vel[0]
    base_vel[1] = -base_vel[1]
    base_vel[2] = -base_vel[2]

    for i in range(N):
        dt = dt_arr[i]

        # R_jk = Exp(-bias_g * dt)
        omega = np.empty(3)
        omega[0] = -bias_g[0] * dt
        omega[1] = -bias_g[1] * dt
        omega[2] = -bias_g[2] * dt
        R_jk = exp_so3_jit(omega)

        # result = R_jk @ point + base_vel * dt
        p = mat3_vec3_mul(R_jk, points_xyz[i])
        result[i, 0] = p[0] + base_vel[0] * dt
        result[i, 1] = p[1] + base_vel[1] * dt
        result[i, 2] = p[2] + base_vel[2] * dt

    return result


# ─────────────────────────────────────────────────────────────
#  Batch covariance transform (used in state_estimation & pv_update)
# ─────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def batch_transform_covariance_jit(body_cov_all, p_imu_all, R, rot_var, t_var):
    """Compute world-frame covariance for N points.

    cov[i] = R @ body_cov[i] @ R.T + skew(p_imu[i]) @ rot_var @ skew(p_imu[i]).T + t_var

    Args:
        body_cov_all: (N,3,3) body covariances
        p_imu_all: (N,3) points in IMU frame
        R: (3,3) rotation matrix (rot_end or rot_ext)
        rot_var: (3,3) rotation covariance
        t_var: (3,3) translation covariance

    Returns:
        (N,3,3) world-frame covariances
    """
    N = body_cov_all.shape[0]
    result = np.empty((N, 3, 3))
    R_T = mat3_transpose(R)

    for idx in prange(N):
        # R @ body_cov @ R.T
        tmp = mat3_mul(R, body_cov_all[idx])
        cov_body_rot = mat3_mul(tmp, R_T)

        # skew(p_imu) @ rot_var @ skew(p_imu).T
        cross = skew_jit(p_imu_all[idx])
        cross_T = mat3_transpose(cross)
        tmp2 = mat3_mul(cross, rot_var)
        cov_cross = mat3_mul(tmp2, cross_T)

        # Sum
        for i in range(3):
            for j in range(3):
                result[idx, i, j] = cov_body_rot[i, j] + cov_cross[i, j] + t_var[i, j]

    return result


# ─────────────────────────────────────────────────────────────
#  Batch world-frame transform
# ─────────────────────────────────────────────────────────────

@njit(parallel=True, cache=True)
def batch_transform_points_jit(ext_R, ext_T, rot, pos, points_body):
    """Transform points: p_world = rot @ (ext_R @ p_body + ext_T) + pos.

    Also returns p_imu = ext_R @ p_body + ext_T.

    Args:
        ext_R: (3,3)
        ext_T: (3,)
        rot: (3,3)
        pos: (3,)
        points_body: (N,3)

    Returns:
        (pw_all (N,3), p_imu_all (N,3))
    """
    N = points_body.shape[0]
    pw_all = np.empty((N, 3))
    p_imu_all = np.empty((N, 3))

    for idx in prange(N):
        # p_imu = ext_R @ pb + ext_T
        p_imu = mat3_vec3_mul(ext_R, points_body[idx])
        p_imu[0] += ext_T[0]
        p_imu[1] += ext_T[1]
        p_imu[2] += ext_T[2]

        p_imu_all[idx, 0] = p_imu[0]
        p_imu_all[idx, 1] = p_imu[1]
        p_imu_all[idx, 2] = p_imu[2]

        # p_world = rot @ p_imu + pos
        pw = mat3_vec3_mul(rot, p_imu)
        pw_all[idx, 0] = pw[0] + pos[0]
        pw_all[idx, 1] = pw[1] + pos[1]
        pw_all[idx, 2] = pw[2] + pos[2]

    return pw_all, p_imu_all


# ─────────────────────────────────────────────────────────────
#  Full residual list builder using plane cache (JIT)
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def build_residual_list_jit(
    points_body,    # (N,3)
    points_world,   # (N,3)
    cov_all,        # (N,3,3)
    body_cov_all,   # (N,3,3)
    voxel_size,     # float
    sigma_num,      # float
    # Plane cache flat arrays:
    plane_normals,     # (P,3) all planes across all voxels
    plane_d,           # (P,) plane d values
    plane_centers,     # (P,3) plane centers
    plane_vars,        # (P,6,6) plane covariances
    plane_radii,       # (P,) plane radii
    # Voxel-to-plane index mapping:
    voxel_plane_starts, # typed Dict[(i64,i64,i64) → i64] start index
    voxel_plane_counts, # typed Dict[(i64,i64,i64) → i64] count
    # Voxel center and quarter_length for neighbor search:
    voxel_centers_map,  # typed Dict[(i64,i64,i64) → i64] index into voxel_info
    voxel_info_centers, # (V,3) voxel centers
    voxel_info_ql,      # (V,) quarter lengths
):
    """Build the full residual list in JIT-compiled code.

    Returns arrays of matched point data (variable length up to N):
        out_point_b (N,3), out_point_w (N,3), out_normal (N,3),
        out_center (N,3), out_plane_var (N,6,6), out_body_cov (N,3,3),
        out_d (N,), out_dis_to_plane (N,), out_count (scalar)
    """
    N = points_body.shape[0]

    # Pre-allocate output (max N matches)
    out_point_b = np.empty((N, 3))
    out_point_w = np.empty((N, 3))
    out_normal = np.empty((N, 3))
    out_center = np.empty((N, 3))
    out_plane_var = np.empty((N, 6, 6))
    out_body_cov = np.empty((N, 3, 3))
    out_d = np.empty(N)
    out_dis_to_plane = np.empty(N)
    out_count = 0

    # Batch voxel key computation
    inv_vs = 1.0 / voxel_size
    for i in range(N):
        pw = points_world[i]

        # Compute voxel key
        lx = pw[0] * inv_vs
        ly = pw[1] * inv_vs
        lz = pw[2] * inv_vs
        kx = np.int64(lx - 1.0) if lx < 0 else np.int64(lx)
        ky = np.int64(ly - 1.0) if ly < 0 else np.int64(ly)
        kz = np.int64(lz - 1.0) if lz < 0 else np.int64(lz)
        key = (kx, ky, kz)

        best_prob = 0.0
        best_found = False
        best_plane_idx = -1
        best_dis_signed = 0.0

        # Try primary voxel
        if key in voxel_plane_starts:
            start = voxel_plane_starts[key]
            count = voxel_plane_counts[key]
            for pi in range(start, start + count):
                success, this_prob, dis_signed = compute_residual_math(
                    pw, plane_normals[pi], plane_d[pi],
                    plane_centers[pi], plane_vars[pi],
                    cov_all[i], sigma_num, plane_radii[pi])
                if success and this_prob > best_prob:
                    best_found = True
                    best_prob = this_prob
                    best_plane_idx = pi
                    best_dis_signed = dis_signed

        # Try neighbor voxel if primary failed
        if not best_found and key in voxel_centers_map:
            vi = voxel_centers_map[key]
            vc = voxel_info_centers[vi]
            ql = voxel_info_ql[vi]

            nkx = kx
            nky = ky
            nkz = kz
            if lx > vc[0] + ql:
                nkx += 1
            elif lx < vc[0] - ql:
                nkx -= 1
            if ly > vc[1] + ql:
                nky += 1
            elif ly < vc[1] - ql:
                nky -= 1
            if lz > vc[2] + ql:
                nkz += 1
            elif lz < vc[2] - ql:
                nkz -= 1

            near_key = (nkx, nky, nkz)
            if near_key in voxel_plane_starts:
                start = voxel_plane_starts[near_key]
                count = voxel_plane_counts[near_key]
                for pi in range(start, start + count):
                    success, this_prob, dis_signed = compute_residual_math(
                        pw, plane_normals[pi], plane_d[pi],
                        plane_centers[pi], plane_vars[pi],
                        cov_all[i], sigma_num, plane_radii[pi])
                    if success and this_prob > best_prob:
                        best_found = True
                        best_prob = this_prob
                        best_plane_idx = pi
                        best_dis_signed = dis_signed

        if best_found:
            idx = out_count
            for j in range(3):
                out_point_b[idx, j] = points_body[i, j]
                out_point_w[idx, j] = points_world[i, j]
                out_normal[idx, j] = plane_normals[best_plane_idx, j]
                out_center[idx, j] = plane_centers[best_plane_idx, j]
            for j in range(6):
                for k in range(6):
                    out_plane_var[idx, j, k] = plane_vars[best_plane_idx, j, k]
            for j in range(3):
                for k in range(3):
                    out_body_cov[idx, j, k] = body_cov_all[i, j, k]
            out_d[idx] = plane_d[best_plane_idx]
            out_dis_to_plane[idx] = best_dis_signed
            out_count += 1

    return (out_point_b, out_point_w, out_normal, out_center,
            out_plane_var, out_body_cov, out_d, out_dis_to_plane, out_count)


# ─────────────────────────────────────────────────────────────
#  Warm-up function — call once at startup to pre-compile all JIT
# ─────────────────────────────────────────────────────────────

def warmup():
    """Pre-compile all JIT functions with dummy data."""
    # SO(3) primitives
    v = np.array([0.1, 0.2, 0.3])
    skew_jit(v)
    exp_so3_jit(v)
    exp_so3_dt_jit(v, 0.01)

    # Matrix ops
    A = np.eye(3)
    mat3_mul(A, A)
    mat3_vec3_mul(A, v)
    mat3_transpose(A)
    vec3_dot(v, v)
    vec3_norm(v)
    vec3_cross(v, v)

    # Body covariance
    pb = np.array([1.0, 2.0, 3.0])
    calc_body_cov_jit(pb, 0.02, 0.05)
    pb_batch = np.random.randn(4, 3)
    calc_body_cov_batch_jit(pb_batch, 0.02, 0.05)

    # Residual math
    normal = np.array([0.0, 0.0, 1.0])
    center = np.zeros(3)
    plane_var = np.eye(6) * 0.01
    pv_var = np.eye(3) * 0.01
    compute_residual_math(v, normal, 0.0, center, plane_var, pv_var, 3.0, 0.5)

    # Jacobian
    M = 2
    build_jacobian_hth_jit(
        np.random.randn(M, 3), np.random.randn(M, 3),
        np.random.randn(M, 3), np.random.randn(M, 6, 6),
        np.random.randn(M, 3, 3), np.random.randn(M),
        np.eye(3), np.zeros(3), np.eye(3), np.zeros(3), np.eye(3), np.eye(3))

    # Undistortion
    pts = np.random.randn(2, 3)
    dt = np.array([0.01, 0.02])
    undistort_bracket_jit(
        pts, dt, np.eye(3), v, v, v, v, v,
        np.eye(3), v, np.eye(3), v)

    # Forward without IMU
    forward_without_imu_undistort_jit(pts, dt, v, np.eye(3), v)

    # Batch transform
    batch_transform_covariance_jit(
        np.random.randn(2, 3, 3), np.random.randn(2, 3),
        np.eye(3), np.eye(3), np.eye(3))

    batch_transform_points_jit(np.eye(3), np.zeros(3), np.eye(3), np.zeros(3),
                                np.random.randn(2, 3))

    # Plane cache residual builder
    from numba import types
    from numba.typed import Dict
    key_type = types.UniTuple(types.int64, 3)
    starts = Dict.empty(key_type, types.int64)
    counts = Dict.empty(key_type, types.int64)
    vcmap = Dict.empty(key_type, types.int64)
    starts[(np.int64(0), np.int64(0), np.int64(0))] = np.int64(0)
    counts[(np.int64(0), np.int64(0), np.int64(0))] = np.int64(1)
    vcmap[(np.int64(0), np.int64(0), np.int64(0))] = np.int64(0)
    build_residual_list_jit(
        np.random.randn(2, 3), np.random.randn(2, 3),
        np.random.randn(2, 3, 3), np.random.randn(2, 3, 3),
        0.5, 3.0,
        np.random.randn(1, 3), np.random.randn(1),
        np.random.randn(1, 3), np.random.randn(1, 6, 6),
        np.random.randn(1),
        starts, counts,
        vcmap, np.random.randn(1, 3), np.random.randn(1))
