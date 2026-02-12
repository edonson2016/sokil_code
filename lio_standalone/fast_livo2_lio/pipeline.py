"""Main LIO pipeline orchestration.

Matches the ONLY_LIO path through C++ LIVMapper::run(), sync_packages(),
processImu(), handleLIO().
"""
import os
import time
import numpy as np
from tqdm import tqdm

from .config import PipelineConfig
from .state import StatesGroup
from .bag_reader import read_bag
from .preprocess import Preprocessor
from .downsampler import voxel_grid_downsample
from .imu_processor import ImuProcessor
from .voxel_map import VoxelMapManager, calc_body_cov, calc_body_cov_batch
from .output import (rotation_matrix_to_quaternion, write_tum,
                     write_odometry_csv, write_scan_csv)
from .so3 import skew, skew_batch
from .types import ImuData, LidarScan
from .numba_kernels import (
    calc_body_cov_batch_jit,
    batch_transform_covariance_jit,
    warmup as numba_warmup,
)


class LIOPipeline:
    """Offline LiDAR-Inertial Odometry pipeline.

    Reads a rosbag, processes all LiDAR+IMU data, and outputs a TUM trajectory.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config

        self.preprocessor = Preprocessor(
            blind=config.blind,
            point_filter_num=config.point_filter_num,
        )

        self.imu_processor = ImuProcessor(
            acc_cov=config.acc_cov,
            gyr_cov=config.gyr_cov,
            max_init_count=config.imu_int_frame,
            ext_R=config.extrinsic_R,
            ext_T=config.extrinsic_T,
            ba_bg_est_en=config.ba_bg_est_en,
            gravity_est_en=config.gravity_est_en,
            exposure_estimate_en=config.exposure_estimate_en,
            imu_en=config.imu_en,
        )

        vc = config.voxel
        self.voxel_manager = VoxelMapManager(
            voxel_size=vc.max_voxel_size,
            max_layer=vc.max_layer,
            max_iterations=vc.max_iterations,
            planer_threshold=vc.planner_threshold,
            sigma_num=vc.sigma_num,
            dept_err=vc.dept_err,
            beam_err=vc.beam_err,
            max_points_num=vc.max_points_num,
            layer_init_num=vc.layer_init_num,
            ext_R=config.extrinsic_R,
            ext_T=config.extrinsic_T,
        )

        self.state = StatesGroup()
        self.ext_R = config.extrinsic_R
        self.ext_T = config.extrinsic_T
        self.filter_size = config.filter_size_surf
        self.lidar_map_inited = False
        self.trajectory = []

    def _transform_lidar(self, rot, pos, points_body):
        """Transform points from body to world frame.

        Matches C++ TransformLidar (voxel_map.cpp:513-530):
          p_w = rot @ (extR @ p_b + extT) + pos
        """
        # Vectorized: (N,3)
        p_imu = (self.ext_R @ points_body.T).T + self.ext_T  # (N, 3)
        p_world = (rot @ p_imu.T).T + pos  # (N, 3)
        return p_world

    def run(self, bag_path: str, output_path: str, scan_output_dir: str = None):
        """Process an entire bag file and write the trajectory.

        Args:
            bag_path: Path to the .bag file.
            output_path: Path for the output trajectory file (.csv or .txt).
            scan_output_dir: If set, save per-scan world-frame point clouds
                             as CSV files in this directory.
        """
        if scan_output_dir:
            os.makedirs(scan_output_dir, exist_ok=True)
        # Pre-compile all Numba JIT functions
        print("[Pipeline] Compiling Numba JIT kernels...")
        numba_warmup()
        print("[Pipeline] JIT compilation complete.")

        print(f"[Pipeline] Reading bag: {bag_path}")
        print(f"[Pipeline] LiDAR topic: {self.config.lid_topic}")
        print(f"[Pipeline] IMU topic: {self.config.imu_topic}")
        print(f"[Pipeline] IMU enabled: {self.config.imu_en}")

        # Phase 1: Read and buffer all messages
        lidar_scans = []
        imu_msgs = []

        for msg_type, data in read_bag(
            bag_path,
            self.config.lid_topic,
            self.config.imu_topic,
            lidar_time_offset=self.config.lidar_time_offset,
            imu_time_offset=self.config.imu_time_offset,
        ):
            if msg_type == 'lidar':
                scan = data
                # Preprocess
                pts, inten, times = self.preprocessor.process(
                    scan.points, scan.intensities, scan.timestamps_ms
                )
                if len(pts) > 1:
                    scan.points = pts
                    scan.intensities = inten
                    scan.timestamps_ms = times
                    lidar_scans.append(scan)
            elif msg_type == 'imu':
                imu_msgs.append(data)

        print(f"\n[Pipeline] Loaded {len(lidar_scans)} LiDAR scans, "
              f"{len(imu_msgs)} IMU messages")

        if len(lidar_scans) == 0:
            print("[Pipeline] No LiDAR scans found. Check topic name.")
            return

        # Phase 2: Process sequentially
        imu_idx = 0
        scan_count = 0
        t_start = time.time()

        pbar = tqdm(enumerate(lidar_scans), total=len(lidar_scans),
                    desc="Processing scans", unit="scan", dynamic_ncols=True)
        for scan_num, scan in pbar:
            scan_end_time = (scan.header_time +
                             scan.timestamps_ms[-1] / 1000.0)

            # Collect IMU messages up to scan end time
            scan_imu = []
            while (imu_idx < len(imu_msgs) and
                   imu_msgs[imu_idx].timestamp <= scan_end_time):
                scan_imu.append(imu_msgs[imu_idx])
                imu_idx += 1

            if self.config.imu_en:
                # IMU initialization phase
                if self.imu_processor.imu_need_init:
                    if len(scan_imu) == 0:
                        continue
                    done = self.imu_processor.imu_init(scan_imu, self.state)
                    if not done:
                        continue
                    tqdm.write(f"[Pipeline] IMU initialized. Gravity: "
                              f"{self.state.gravity}")
                    continue

                # Forward propagation and undistortion
                if len(scan_imu) == 0:
                    continue

                imu_poses = self.imu_processor.forward_propagation(
                    scan_imu, self.state, scan_end_time
                )

                undistorted = self.imu_processor.undistort_points(
                    scan.points, scan.timestamps_ms,
                    self.state, imu_poses,
                    scan.timestamps_ms[-1]
                )
            else:
                # No IMU: simple forward without IMU
                undistorted = self.imu_processor.forward_without_imu(
                    self.state, scan.points, scan.timestamps_ms,
                    scan.header_time
                )

            # Downsample
            down_body = voxel_grid_downsample(undistorted, self.filter_size)
            feats_down_size = len(down_body)

            if feats_down_size < 1:
                continue

            # Transform to world frame
            down_world = self._transform_lidar(
                self.state.rot_end, self.state.pos_end, down_body
            )

            if not self.lidar_map_inited:
                # First frame: build initial map
                self.lidar_map_inited = True
                self.voxel_manager.build_voxel_map(
                    down_body, down_world, self.state
                )
                world_points = down_world
                tqdm.write(f"[Pipeline] Voxel map initialized with "
                          f"{len(self.voxel_manager.voxel_map)} voxels")
            else:
                # State estimation
                state_propagat = self.state.copy()
                self.state = self.voxel_manager.state_estimation(
                    self.state, state_propagat, down_body
                )

                # Update map with new points (re-transform with updated state)
                down_world_updated = self._transform_lidar(
                    self.state.rot_end, self.state.pos_end, down_body
                )
                world_points = down_world_updated

                # Compute covariances for map update (JIT parallel)
                rot_ext = self.state.rot_end @ self.ext_R
                rot_cov = self.state.cov[0:3, 0:3]
                t_cov = self.state.cov[3:6, 3:6]

                body_cov_all = calc_body_cov_batch_jit(
                    down_body, self.voxel_manager.dept_err,
                    self.voxel_manager.beam_err)  # (N,3,3)
                p_imu_all = (self.ext_R @ down_body.T).T + self.ext_T  # (N,3)

                var_all = batch_transform_covariance_jit(
                    body_cov_all, p_imu_all, rot_ext, rot_cov, t_cov)  # (N,3,3)

                pv_update = [(down_world_updated[i].copy(), var_all[i])
                             for i in range(feats_down_size)]

                self.voxel_manager.update_voxel_map(pv_update)

            # Record pose
            q = rotation_matrix_to_quaternion(self.state.rot_end)
            self.trajectory.append((
                scan_end_time,
                self.state.pos_end.copy(),
                q,
            ))

            # Save per-scan point cloud CSV
            if scan_output_dir:
                scan_csv = os.path.join(
                    scan_output_dir, f"scan_{scan_count:06d}.csv")
                write_scan_csv(scan_csv, world_points)

            scan_count += 1
            elapsed = time.time() - t_start
            rate = scan_count / elapsed if elapsed > 0 else 0
            pbar.set_postfix(rate=f"{rate:.1f} scans/s",
                             voxels=len(self.voxel_manager.voxel_map))

        pbar.close()

        # Phase 3: Write output
        elapsed = time.time() - t_start
        print(f"\n[Pipeline] Done. {scan_count} scans in {elapsed:.1f}s "
              f"({scan_count / elapsed:.1f} scans/s)")

        if output_path.endswith('.csv'):
            write_odometry_csv(output_path, self.trajectory)
        else:
            write_tum(output_path, self.trajectory)
        print(f"[Pipeline] Trajectory written to: {output_path}")
