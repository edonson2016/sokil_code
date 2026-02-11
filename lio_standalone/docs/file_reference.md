# File Reference

Quick-reference for every file in the codebase, what it does, and what calls it.

---

## Top-Level Files

### `run.py`
**The main entry point.** Orchestrates the three-phase pipeline:
1. Runs LIO (Phase 1) via `LIOPipeline`
2. Builds world point cloud (Phase 2) by re-reading the bag and applying odometry
3. Generates an interactive HTML viewer (Phase 3) with Three.js

Also handles CLI argument parsing and interactive bag file selection from `Scans/`.

**Key functions:**
- `find_bag_files(scans_dir)` — recursively finds `.bag` files
- `prompt_bag_selection(scans_dir)` — interactive bag chooser
- `build_world_pointcloud(bag_path, odom_csv, config, output_csv)` — Phase 2
- `generate_html_viewer(world_pts, world_int, output_html)` — Phase 3
- `_build_viewer_html(meta, b64_data, total_points)` — HTML template generation
- `main()` — CLI entry point

### `avia.yaml`
Default configuration for a Livox Avia LiDAR setup. Defines topics, extrinsics,
preprocessing parameters, IMU settings, and voxel map tuning.

---

## `fast_livo2_lio/` Package

### `__init__.py`
Empty init; makes the directory a Python package.

### `types.py`
**Data structures** used throughout the pipeline. All are `@dataclass`:
- `LidarScan` — one LiDAR sweep (points, intensities, timestamps)
- `ImuData` — one IMU reading (timestamp, acceleration, angular velocity)
- `Pose6D` — full pose at one instant (rotation, position, velocity, acceleration)
- `PointWithVar` — point in body/world frames with covariance
- `PointToPlane` — matched point-to-plane residual for EKF

### `config.py`
**Configuration loader.** Reads YAML and populates `PipelineConfig` and
`VoxelMapConfig` dataclasses. Supports the same key paths as the C++ FAST-LIVO2
config format.

**Called by:** `run.py` → `load_config(yaml_path)`

### `bag_reader.py`
**ROS1 bag parser.** Uses `rosbags` library (no ROS needed).
- Parses Livox `CustomMsg` (binary protocol, 19 bytes per point)
- Parses standard `PointCloud2` messages
- Parses `sensor_msgs/Imu` messages
- Auto-detects per-point time fields for different LiDAR vendors
- Returns messages sorted chronologically

**Called by:** `pipeline.py` → `read_bag()`

### `preprocess.py`
**Point cloud cleanup.** Removes near-field points (blind zone), decimates,
removes NaN/Inf, sorts by timestamp. All vectorized NumPy.

**Called by:** `pipeline.py` → `Preprocessor.process()`

### `downsampler.py`
**Voxel grid downsampling.** Quantizes 3D points into voxels, returns centroid
per voxel. Pure NumPy implementation.

**Called by:** `pipeline.py` → `voxel_grid_downsample()`

### `imu_processor.py`
**IMU processing engine.** Three modes:
1. `imu_init()` — gravity + bias initialization from first N readings
2. `forward_propagation()` — integrates IMU, propagates state + covariance
3. `undistort_points()` — motion compensation per point using IMU poses

Also has `forward_without_imu()` for LiDAR-only operation.

**Called by:** `pipeline.py` → `ImuProcessor`

### `state.py`
**19-dim EKF state.** Implements `boxplus` (state + delta) and `boxminus`
(state - state) using SO(3) exponential/logarithm maps. Carries pose,
velocity, biases, gravity, and a 19x19 covariance matrix.

**Called by:** `pipeline.py`, `imu_processor.py`, `voxel_map.py`

### `so3.py`
**SO(3) Lie group math.** Rodrigues formula implementations:
- `exp_so3(v)` — angle-axis to rotation matrix
- `log_so3(R)` — rotation matrix to angle-axis
- `skew(v)` — vector to skew-symmetric matrix
- Batch versions for vectorized processing

**Called by:** `state.py`, `imu_processor.py`, `voxel_map.py`

### `numba_kernels.py`
**JIT-compiled performance kernels.** All marked `@njit` (Numba no-Python mode):
- SO(3) primitives (compiled versions of `so3.py` functions)
- 3x3 matrix math (multiply, transpose, dot, cross)
- `calc_body_cov_jit` — per-point sensor covariance
- `compute_residual_math` — point-to-plane distance + probability
- `build_jacobian_hth_jit` — full Jacobian accumulation in one pass
- `undistort_bracket_jit` — motion compensation per IMU bracket
- `batch_transform_covariance_jit` — parallel covariance propagation
- `build_residual_list_jit` — full residual list with plane cache lookup

`warmup()` pre-compiles everything with dummy data at startup.

**Called by:** `pipeline.py`, `imu_processor.py`, `voxel_map.py`

### `voxel_map.py`
**The mapping and localization core.** Three main classes:

- **`VoxelPlane`** — fitted plane (normal, center, eigenvalues, covariance)
- **`VoxelOctoTree`** — octree node with PCA plane fitting and hierarchical
  subdivision. Each leaf stores points and fits planes when enough accumulate.
- **`VoxelMapManager`** — top-level hash map of voxels to octrees.
  - `build_voxel_map()` — first-frame initialization
  - `update_voxel_map()` — incremental point insertion
  - `state_estimation()` — iterative EKF: transforms points, finds plane
    correspondences, builds Jacobians, solves for state update
  - `_rebuild_plane_cache()` — flattens all plane data into arrays for JIT

**Called by:** `pipeline.py` → `VoxelMapManager`

### `output.py`
**File writers** for trajectory and point cloud output:
- `write_odometry_csv()` — CSV with header row
- `write_tum()` — TUM benchmark format (space-separated)
- `write_scan_csv()` — per-scan point cloud with optional intensity

**Called by:** `pipeline.py`, `run.py`

---

## Output Files

| File | Format | Contents |
|------|--------|----------|
| `odometry.csv` | CSV | `timestamp, tx, ty, tz, qx, qy, qz, qw` per scan |
| `raw_scans/scan_NNNNNN.csv` | CSV | `x, y, z` per point (world frame, one scan) |
| `world_pointcloud.csv` | CSV | `x, y, z, intensity, timestamp` for all points |
| `viewer.html` | HTML | Self-contained Three.js 3D viewer |

---

## Call Graph Summary

```
run.py main()
  ├── prompt_bag_selection()        # if no bag arg
  ├── load_config()                 # config.py
  ├── LIOPipeline(config)           # pipeline.py
  │     ├── Preprocessor            # preprocess.py
  │     ├── ImuProcessor            # imu_processor.py
  │     │     └── numba_kernels     # undistort_bracket_jit, etc.
  │     ├── VoxelMapManager         # voxel_map.py
  │     │     ├── VoxelOctoTree     #   octree + plane fitting
  │     │     └── numba_kernels     #   calc_body_cov, residuals, jacobians
  │     └── read_bag()              # bag_reader.py
  ├── build_world_pointcloud()      # run.py (re-reads bag + applies odom)
  └── generate_html_viewer()        # run.py (Three.js HTML output)
```
