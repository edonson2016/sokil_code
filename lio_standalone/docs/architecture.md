# FAST-LIVO2 LIO Standalone — Architecture Guide

## Overview

This is a self-contained Python port of the LiDAR-Inertial Odometry (LIO) subsystem
from [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2). It processes ROS1 bag
files containing Livox LiDAR and IMU data and produces registered 3D point clouds
with an interactive HTML viewer — all without needing a ROS installation.

```
┌──────────────┐
│  .bag file   │   ROS1 bag with LiDAR + IMU topics
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 1: LiDAR-Inertial Odometry  (pipeline.py)        │
│                                                          │
│  bag_reader ──► preprocess ──► IMU fwd propagation       │
│                                  + undistortion          │
│                 ──► voxel downsample ──► voxel map       │
│                                  state estimation (EKF)  │
│                                                          │
│  Output: odometry.csv  (6-DOF trajectory per scan)       │
│          raw_scans/     (per-scan body-frame CSVs)       │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 2: World Point Cloud  (run.py)                    │
│                                                          │
│  Re-reads bag, matches each scan to its odometry pose,   │
│  transforms all points into a shared world frame.        │
│                                                          │
│  Output: world_pointcloud.csv  (x, y, z, intensity, t)  │
└──────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│  Phase 3: HTML Viewer  (run.py)                          │
│                                                          │
│  Downsamples, base64-encodes, embeds in a self-contained │
│  Three.js HTML file with orbit controls and color modes. │
│                                                          │
│  Output: viewer.html                                     │
└──────────────────────────────────────────────────────────┘
```

---

## Entry Point

**`run.py`** — the single command to run the full pipeline.

```bash
# Interactive bag selection (scans Scans/ folder):
python3 run.py

# Direct bag path:
python3 run.py path/to/scan.bag

# With options:
python3 run.py scan.bag --config custom.yaml --output-dir results/
```

When run without arguments, it searches the `Scans/` folder for `.bag` files and
prompts the user to select one.

---

## Module Reference

All pipeline modules live in `fast_livo2_lio/`.

### `types.py` — Data Structures

| Class | Purpose |
|-------|---------|
| `LidarScan` | A single LiDAR sweep: `points (N,3)`, `intensities (N,)`, `timestamps_ms (N,)`, `header_time` |
| `ImuData` | A single IMU measurement: `timestamp`, `acc (3,)`, `gyr (3,)` |
| `Pose6D` | IMU pose at a point in time: position, rotation, velocity, acceleration, angular velocity |
| `PointWithVar` | Point with body/world coordinates and covariance matrix |
| `PointToPlane` | Point-to-plane measurement residual for EKF update |

### `config.py` — Configuration

Reads YAML config files (e.g., `avia.yaml`) using the same key layout as the
original C++ FAST-LIVO2 configs. Returns a `PipelineConfig` dataclass.

Key parameters:
- `lid_topic` / `imu_topic` — ROS topic names
- `blind` — minimum point range in meters (filters near-field noise)
- `point_filter_num` — keep every Nth point (decimation)
- `filter_size_surf` — voxel grid downsample leaf size
- `extrinsic_R`, `extrinsic_T` — LiDAR-to-IMU rigid transform
- `voxel` — nested `VoxelMapConfig` with octree/plane fitting settings

### `bag_reader.py` — ROS Bag Parsing

Uses the `rosbags` library (pure Python, no ROS install). Supports:
- **Livox CustomMsg** — Livox Avia native format with per-point `offset_time`
- **Standard PointCloud2** — generic ROS point cloud format
- **sensor_msgs/Imu** — standard IMU messages

Auto-detects per-point time fields across LiDAR vendors (Velodyne `time`,
Ouster `t`, Hesai `timestamp`, Livox `offset_time`).

Returns all messages sorted chronologically as `('lidar', LidarScan)` or
`('imu', ImuData)` tuples.

### `preprocess.py` — Point Cloud Preprocessing

Vectorized NumPy operations:
1. **Blind zone filter** — removes points within `blind` meters of origin
2. **Point decimation** — keeps every Nth point
3. **NaN/Inf removal** — discards invalid readings
4. **Time sort** — sorts points by per-point timestamp offset

### `downsampler.py` — Voxel Grid Downsampling

Replaces PCL's `VoxelGrid`. Quantizes points into cubic voxels of a given
`leaf_size` and returns the centroid of each occupied voxel. Reduces point
density for efficient mapping.

### `imu_processor.py` — IMU Processing

Three core operations:

1. **Initialization** (`imu_init`) — averages the first N IMU readings to
   estimate gravity direction and initial biases.

2. **Forward propagation** (`forward_propagation`) — integrates IMU
   measurements (gyroscope + accelerometer) between scans, propagating the
   state (position, velocity, rotation) and the 19x19 covariance matrix.

3. **Point undistortion** (`undistort_points`) — corrects for ego-motion
   during a LiDAR sweep. Each point is back-projected to the scan-end frame
   using interpolated IMU poses, eliminating motion blur. Uses Numba JIT.

Also supports an IMU-less mode (`forward_without_imu`) using constant-velocity
assumptions.

### `state.py` — EKF State

19-dimensional state vector with SO(3) boxplus/boxminus semantics:

| Index | Component | Description |
|-------|-----------|-------------|
| 0-2 | rotation | SO(3) via Exp/Log maps |
| 3-5 | position | xyz in world frame |
| 6 | inv_expo_time | inverse exposure time |
| 7-9 | velocity | xyz in world frame |
| 10-12 | bias_g | gyroscope bias |
| 13-15 | bias_a | accelerometer bias |
| 16-18 | gravity | gravity vector |

### `so3.py` — SO(3) Math

Rodrigues-formula implementations:
- `exp_so3(v)` — angle-axis vector to rotation matrix
- `log_so3(R)` — rotation matrix to angle-axis vector
- `skew(v)` — 3-vector to skew-symmetric matrix
- Batch versions for vectorized operations

### `voxel_map.py` — Voxel Map + State Estimation

The core mapping and localization engine:

- **VoxelOctoTree** — each voxel contains an octree that subdivides up to
  `max_layer` levels. At each leaf, PCA fits a plane to the accumulated
  points. Plane quality is measured by eigenvalue ratio.

- **VoxelMapManager** — hash table mapping `(i, j, k)` voxel keys to octrees.
  - `build_voxel_map()` — initializes the map from the first frame
  - `update_voxel_map()` — incrementally adds new points
  - `state_estimation()` — iterative EKF using point-to-plane residuals
  - Uses a "plane cache" (flat arrays + Numba typed dicts) to enable
    JIT-compiled residual building without Python overhead

- **State estimation loop** (per scan):
  1. Transform body points to world frame with current state
  2. Compute per-point covariances (sensor model + pose uncertainty)
  3. Find point-to-plane correspondences via voxel lookup
  4. Build Jacobian H, solve H^T R^{-1} H delta = H^T R^{-1} z
  5. Apply state update (boxplus), check convergence
  6. Repeat up to `max_iterations` times

### `numba_kernels.py` — JIT-Compiled Kernels

Performance-critical inner loops compiled to machine code via Numba `@njit`:
- SO(3) operations (skew, exp, matrix multiply)
- Body covariance computation (parallel batch)
- Point-to-plane residual math
- Jacobian H^T H accumulation
- Point undistortion (per-bracket transform)
- World-frame covariance transform (parallel batch)

All kernels are pre-compiled on startup via `warmup()`.

### `output.py` — File Writers

- `write_odometry_csv()` — TUM-format trajectory: `timestamp, tx, ty, tz, qx, qy, qz, qw`
- `write_tum()` — space-separated TUM format
- `write_scan_csv()` — per-scan point cloud with optional intensity

---

## Data Flow for a Single Scan

```
Raw bag message
       │
       ▼
  Parse (bag_reader) ──► LidarScan {points, intensities, timestamps_ms}
       │
       ▼
  Preprocess ──► blind filter + decimate + sort by time
       │
       ▼
  Collect IMU messages up to scan end time
       │
       ▼
  IMU forward propagation ──► state (R, p, v, biases, gravity)
       │                       + covariance P
       ▼
  Undistort points using interpolated IMU poses
       │
       ▼
  Voxel downsample (centroid per voxel)
       │
       ▼
  Transform body ──► world frame: p_w = R @ (R_ext @ p_b + t_ext) + t
       │
       ▼
  First scan? ──► Build voxel map
  Later scan? ──► State estimation (iterative EKF)
                  ──► Update voxel map with new points
       │
       ▼
  Record (timestamp, position, quaternion) to trajectory
```

---

## Configuration (avia.yaml)

The YAML config mirrors the C++ FAST-LIVO2 format. Key sections:

| Section | Key Parameters |
|---------|---------------|
| `common` | `lid_topic`, `imu_topic` |
| `extrin_calib` | `extrinsic_T [3]`, `extrinsic_R [9]` — LiDAR-to-IMU transform |
| `preprocess` | `blind`, `point_filter_num`, `filter_size_surf`, `lidar_type` |
| `imu` | `imu_en`, `acc_cov`, `gyr_cov`, `imu_int_frame` |
| `lio` | `voxel_size`, `max_layer`, `max_iterations`, `dept_err`, `beam_err` |
| `local_map` | `map_sliding_en`, `half_map_size` |

---

## Dependencies

- `numpy` — array math
- `scipy` — quaternion conversion (`Rotation`)
- `numba` — JIT compilation for inner loops
- `rosbags` — ROS1 bag reading without ROS
- `tqdm` — progress bars
- `pyyaml` — config loading

No ROS installation is required.
