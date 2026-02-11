"""IMU initialization, forward propagation, covariance propagation, and
point cloud undistortion.

Matches C++ ImuProcess class from src/IMU_Processing.cpp exactly.
"""
import numpy as np
from .so3 import skew, exp_so3, exp_so3_dt, exp_so3_batch
from .state import StatesGroup, DIM_STATE, G_m_s2
from .types import Pose6D, ImuData
from .numba_kernels import undistort_bracket_jit, forward_without_imu_undistort_jit


class ImuProcessor:
    """IMU processing: initialization, state propagation, and point undistortion."""

    def __init__(self, acc_cov: float = 0.1, gyr_cov: float = 0.1,
                 bias_acc_cov: float = 0.0001, bias_gyr_cov: float = 0.0001,
                 inv_expo_cov: float = 0.2,
                 max_init_count: int = 30,
                 ext_R: np.ndarray = None, ext_T: np.ndarray = None,
                 ba_bg_est_en: bool = True,
                 gravity_est_en: bool = True,
                 exposure_estimate_en: bool = False,
                 imu_en: bool = True):
        """
        Args:
            acc_cov: Accelerometer noise covariance scale.
            gyr_cov: Gyroscope noise covariance scale.
            bias_acc_cov: Accelerometer bias random walk covariance.
            bias_gyr_cov: Gyroscope bias random walk covariance.
            inv_expo_cov: Inverse exposure time covariance.
            max_init_count: Number of IMU samples for initialization.
            ext_R: (3, 3) LiDAR-to-IMU rotation (Lid_rot_to_IMU).
            ext_T: (3,) LiDAR-to-IMU translation (Lid_offset_to_IMU).
            ba_bg_est_en: Enable bias estimation.
            gravity_est_en: Enable gravity estimation.
            exposure_estimate_en: Enable exposure time estimation.
            imu_en: Enable IMU processing.
        """
        # Noise covariances (stored as 3-vectors for diagonal construction)
        self.cov_acc = np.array([acc_cov, acc_cov, acc_cov])
        self.cov_gyr = np.array([gyr_cov, gyr_cov, gyr_cov])
        self.cov_bias_gyr = np.array([bias_gyr_cov] * 3)
        self.cov_bias_acc = np.array([bias_acc_cov] * 3)
        self.cov_inv_expo = inv_expo_cov

        # Extrinsics
        self.Lid_rot_to_IMU = ext_R if ext_R is not None else np.eye(3)
        self.Lid_offset_to_IMU = ext_T if ext_T is not None else np.zeros(3)

        # Estimation flags
        self.ba_bg_est_en = ba_bg_est_en
        self.gravity_est_en = gravity_est_en
        self.exposure_estimate_en = exposure_estimate_en
        self.imu_en = imu_en

        # Initialization state
        self.max_init_count = max_init_count
        self.init_iter_num = 1
        self.imu_need_init = True
        self.b_first_frame = True
        self.mean_acc = np.array([0.0, 0.0, -1.0])
        self.mean_gyr = np.zeros(3)
        self.IMU_mean_acc_norm = G_m_s2

        # Last IMU state (carried between scans)
        self.last_imu = ImuData()
        self.angvel_last = np.zeros(3)
        self.acc_s_last = np.zeros(3)
        self.last_prop_end_time = 0.0
        self.imu_time_init = False

    def imu_init(self, imu_list: list, state: StatesGroup) -> bool:
        """Initialize gravity direction from mean accelerometer readings.

        Matches C++ IMU_init (IMU_Processing.cpp:104-149).

        Args:
            imu_list: List of ImuData for this measurement group.
            state: State to initialize.

        Returns:
            True if initialization is complete.
        """
        if not imu_list:
            return False

        if self.b_first_frame:
            self.mean_acc = imu_list[0].acc.copy()
            self.mean_gyr = imu_list[0].gyr.copy()
            self.init_iter_num = 1
            self.b_first_frame = False

        for imu in imu_list:
            self.mean_acc += (imu.acc - self.mean_acc) / self.init_iter_num
            self.mean_gyr += (imu.gyr - self.mean_gyr) / self.init_iter_num
            self.init_iter_num += 1

        self.IMU_mean_acc_norm = np.linalg.norm(self.mean_acc)
        state.gravity = -self.mean_acc / self.IMU_mean_acc_norm * G_m_s2
        state.rot_end = np.eye(3)
        state.bias_g = np.zeros(3)

        self.last_imu = imu_list[-1]

        if self.init_iter_num > self.max_init_count:
            self.imu_need_init = False
            return True
        return False

    def forward_propagation(self, imu_list: list, state: StatesGroup,
                            prop_end_time: float):
        """Forward propagation through IMU measurements with covariance.

        Matches C++ UndistortPcl forward propagation (lines 297-438).

        Args:
            imu_list: List of ImuData for this scan's measurement group.
            state: Current state (modified in place).
            prop_end_time: Target propagation end time.

        Returns:
            List of Pose6D at each IMU measurement time.
        """
        prop_beg_time = self.last_prop_end_time

        # Prepend last IMU from previous scan
        v_imu = [self.last_imu] + list(imu_list)

        # Initialize propagation variables from current state
        acc_imu = self.acc_s_last.copy()
        angvel_avr = self.angvel_last.copy()
        vel_imu = state.vel_end.copy()
        pos_imu = state.pos_end.copy()
        R_imu = state.rot_end.copy()

        # Exposure time
        if not self.imu_time_init:
            tau = 1.0
            self.imu_time_init = True
        else:
            tau = state.inv_expo_time

        imu_poses = []
        # Initial pose
        imu_poses.append(Pose6D(
            offset_time=0.0,
            acc=self.acc_s_last.copy(),
            gyr=self.angvel_last.copy(),
            vel=state.vel_end.copy(),
            pos=state.pos_end.copy(),
            rot=state.rot_end.copy(),
        ))

        Eye3 = np.eye(3)

        for i in range(len(v_imu) - 1):
            head = v_imu[i]
            tail = v_imu[i + 1]

            if tail.timestamp < prop_beg_time:
                continue

            # Average angular velocity and acceleration
            angvel_avr = 0.5 * (head.gyr + tail.gyr)
            acc_avr = 0.5 * (head.acc + tail.acc)

            # Remove biases
            angvel_avr = angvel_avr - state.bias_g
            acc_avr = acc_avr * G_m_s2 / self.IMU_mean_acc_norm - state.bias_a

            # Compute dt and offset time
            if head.timestamp < prop_beg_time:
                dt = tail.timestamp - self.last_prop_end_time
                offs_t = tail.timestamp - prop_beg_time
            elif i != len(v_imu) - 2:
                dt = tail.timestamp - head.timestamp
                offs_t = tail.timestamp - prop_beg_time
            else:
                dt = prop_end_time - head.timestamp
                offs_t = prop_end_time - prop_beg_time

            if dt <= 0:
                continue

            # Covariance propagation matrices
            Exp_f = exp_so3_dt(angvel_avr, dt)
            acc_avr_skew = skew(acc_avr)

            F_x = np.eye(DIM_STATE)
            cov_w = np.zeros((DIM_STATE, DIM_STATE))

            F_x[0:3, 0:3] = exp_so3_dt(angvel_avr, -dt)
            if self.ba_bg_est_en:
                F_x[0:3, 10:13] = -Eye3 * dt
            F_x[3:6, 7:10] = Eye3 * dt
            F_x[7:10, 0:3] = -R_imu @ acc_avr_skew * dt
            if self.ba_bg_est_en:
                F_x[7:10, 13:16] = -R_imu * dt
            if self.gravity_est_en:
                F_x[7:10, 16:19] = Eye3 * dt

            if self.exposure_estimate_en:
                cov_w[6, 6] = self.cov_inv_expo * dt * dt
            cov_w[0:3, 0:3] = np.diag(self.cov_gyr) * dt * dt
            cov_w[7:10, 7:10] = R_imu @ np.diag(self.cov_acc) @ R_imu.T * dt * dt
            cov_w[10:13, 10:13] = np.diag(self.cov_bias_gyr) * dt * dt
            cov_w[13:16, 13:16] = np.diag(self.cov_bias_acc) * dt * dt

            state.cov = F_x @ state.cov @ F_x.T + cov_w

            # State propagation
            R_imu = R_imu @ Exp_f
            acc_imu = R_imu @ acc_avr + state.gravity
            pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt
            vel_imu = vel_imu + acc_imu * dt

            # Save for next iteration
            angvel_avr_save = angvel_avr.copy()
            acc_imu_save = acc_imu.copy()
            self.angvel_last = angvel_avr_save
            self.acc_s_last = acc_imu_save

            imu_poses.append(Pose6D(
                offset_time=offs_t,
                acc=acc_imu_save,
                gyr=angvel_avr_save,
                vel=vel_imu.copy(),
                pos=pos_imu.copy(),
                rot=R_imu.copy(),
            ))

        # Update state
        state.vel_end = vel_imu
        state.rot_end = R_imu
        state.pos_end = pos_imu
        state.inv_expo_time = tau

        # Save for next scan
        self.last_imu = v_imu[-1]
        self.last_prop_end_time = prop_end_time

        return imu_poses

    def undistort_points(self, points_xyz: np.ndarray, times_ms: np.ndarray,
                         state: StatesGroup, imu_poses: list,
                         scan_end_offset_ms: float) -> np.ndarray:
        """Back-project each point to the scan-end frame using IMU poses.

        Matches C++ UndistortPcl backward propagation (lines 494-538).

        Args:
            points_xyz: (N, 3) points in LiDAR body frame.
            times_ms: (N,) per-point time offsets in ms from scan start.
            state: State at scan end.
            imu_poses: List of Pose6D from forward propagation.
            scan_end_offset_ms: Time offset of last point in ms.

        Returns:
            (N, 3) undistorted points in LiDAR body frame at scan-end pose.
        """
        if len(points_xyz) == 0 or len(imu_poses) < 2:
            return points_xyz.copy()

        N = len(points_xyz)
        result = points_xyz.copy()

        # Precompute constants
        extR_Ri = self.Lid_rot_to_IMU.T @ state.rot_end.T
        exrR_extT = self.Lid_rot_to_IMU.T @ self.Lid_offset_to_IMU

        # Get IMU pose offset times for bracket lookup
        pose_offsets = np.array([p.offset_time for p in imu_poses])

        # Convert point times to seconds
        times_s = times_ms / 1000.0

        # For each IMU pose bracket (working backward)
        for kp_idx in range(len(imu_poses) - 1, 0, -1):
            head = imu_poses[kp_idx - 1]
            tail = imu_poses[kp_idx]

            R_head = head.rot
            acc_head = head.acc
            vel_head = head.vel
            pos_head = head.pos
            gyr_head = head.gyr

            # Find points in this bracket: head.offset_time < point_time <= tail.offset_time
            if kp_idx == len(imu_poses) - 1:
                mask = times_s > head.offset_time
            else:
                mask = (times_s > head.offset_time) & (times_s <= tail.offset_time)

            if not np.any(mask):
                continue

            pts_in_bracket = np.ascontiguousarray(points_xyz[mask])
            ts_in_bracket = times_s[mask]

            # JIT-compiled per-bracket undistortion
            dt_arr = ts_in_bracket - head.offset_time  # (M,)

            compensated = undistort_bracket_jit(
                pts_in_bracket, dt_arr, R_head, gyr_head,
                vel_head, pos_head, acc_head, state.pos_end,
                self.Lid_rot_to_IMU, self.Lid_offset_to_IMU,
                extR_Ri, exrR_extT)

            result[mask] = compensated

        return result

    def forward_without_imu(self, state: StatesGroup,
                            points_xyz: np.ndarray, times_ms: np.ndarray,
                            scan_beg_time: float):
        """Process without IMU: simple constant-velocity propagation.

        Matches C++ Forward_without_imu (lines 151-234).

        Args:
            state: Current state (modified in place).
            points_xyz: (N, 3) points in body frame.
            times_ms: (N,) time offsets in ms.
            scan_beg_time: Scan begin time.

        Returns:
            (N, 3) undistorted points.
        """
        if not hasattr(self, 'time_last_scan'):
            dt = 0.1
            self.time_last_scan = scan_beg_time
            self.b_first_frame = False
        else:
            dt = scan_beg_time - self.time_last_scan

        self.time_last_scan = scan_beg_time
        scan_end_offset_s = times_ms[-1] / 1000.0 if len(times_ms) > 0 else 0.0

        # Covariance propagation (constant velocity model)
        F_x = np.eye(DIM_STATE)
        cov_w = np.zeros((DIM_STATE, DIM_STATE))

        Exp_f = exp_so3_dt(state.bias_g, dt)
        F_x[0:3, 0:3] = exp_so3_dt(state.bias_g, -dt)
        F_x[0:3, 10:13] = np.eye(3) * dt
        F_x[3:6, 7:10] = np.eye(3) * dt

        cov_w[10:13, 10:13] = np.diag(self.cov_gyr) * dt * dt
        cov_w[7:10, 7:10] = np.diag(self.cov_acc) * dt * dt

        state.cov = F_x @ state.cov @ F_x.T + cov_w
        state.rot_end = state.rot_end @ Exp_f
        state.pos_end = state.pos_end + state.vel_end * dt

        # Undistort points (JIT)
        result = points_xyz.copy()
        if len(points_xyz) > 0:
            times_s = times_ms / 1000.0
            dt_arr = scan_end_offset_s - times_s  # (N,)

            rot_end_T = np.ascontiguousarray(state.rot_end.T)
            result = forward_without_imu_undistort_jit(
                points_xyz, dt_arr, state.bias_g, rot_end_T, state.vel_end)

        return result
