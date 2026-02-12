# Algorithms Deep Dive

Detailed explanation of the core algorithms in the pipeline.

---

## 1. IMU Initialization

**File:** `imu_processor.py` → `imu_init()`

**Purpose:** Estimate gravity direction from the first N IMU readings while
the sensor is assumed stationary.

**Algorithm:**
1. Compute running mean of accelerometer readings across first `imu_int_frame`
   (default 30) IMU measurements using incremental averaging
2. The mean acceleration vector points opposite to gravity (since the
   accelerometer measures the reaction force to gravity when stationary)
3. Set `gravity = -mean_acc / ||mean_acc|| * 9.81`
4. Store `||mean_acc||` as `IMU_mean_acc_norm` for accelerometer scale
   correction in subsequent processing

After initialization, the state's gravity vector is set and the rotation is
initialized to identity.

---

## 2. IMU Forward Propagation

**File:** `imu_processor.py` → `forward_propagation()`

**Purpose:** Integrate IMU measurements to propagate state between LiDAR scans,
and propagate the error-state covariance for the EKF.

**Algorithm (per IMU pair):**

Given consecutive IMU readings `head` and `tail` with time delta `dt`:

1. **Average angular velocity and acceleration:**
   ```
   omega = 0.5 * (head.gyr + tail.gyr) - bias_g
   a = 0.5 * (head.acc + tail.acc) * G / ||mean_acc|| - bias_a
   ```

2. **State propagation (discrete integration):**
   ```
   R_new = R @ Exp(omega * dt)
   a_world = R_new @ a + gravity
   p_new = p + v * dt + 0.5 * a_world * dt^2
   v_new = v + a_world * dt
   ```

3. **Covariance propagation:**
   ```
   F_x = I + [  -Exp(-omega*dt)  ...  -I*dt    ...            ]
             [       ...         I*dt   ...                    ]
             [  -R@skew(a)*dt    ...         -R*dt   I*dt      ]

   Q = diag(gyr_cov*dt^2, ..., acc_cov*dt^2, bias_gyr_cov*dt^2, bias_acc_cov*dt^2)

   P_new = F_x @ P @ F_x^T + Q
   ```

4. **Record Pose6D** at each IMU timestamp for point undistortion.

---

## 3. Point Undistortion (Motion Compensation)

**File:** `imu_processor.py` → `undistort_points()`,
`numba_kernels.py` → `undistort_bracket_jit()`

**Purpose:** Remove motion blur from LiDAR scans. During a ~100ms sweep,
the sensor moves, so each point was measured at a different pose. This
transforms every point to the scan-end frame.

**Algorithm:**

For each point with time offset `t_point`:

1. Find the IMU pose bracket `[pose_k, pose_{k+1}]` containing `t_point`
2. Interpolate within the bracket:
   ```
   dt = t_point - pose_k.offset_time
   R_i = R_k @ Exp(omega_k * dt)
   p_i = p_k + v_k * dt + 0.5 * a_k * dt^2
   ```
3. Transform point to world frame at its measurement time:
   ```
   p_imu = R_ext @ p_lidar + T_ext
   p_world_at_t = R_i @ p_imu + p_i
   ```
4. Transform back to body frame at scan-end time:
   ```
   p_corrected = R_ext^T @ R_end^T @ (p_world_at_t - p_end) - R_ext^T @ T_ext
   ```

The result: all points appear as if they were measured simultaneously at the
scan-end pose.

---

## 4. Voxel Grid Downsampling

**File:** `downsampler.py` → `voxel_grid_downsample()`

**Purpose:** Reduce point density while preserving geometric structure.

**Algorithm:**
1. Quantize each point to a voxel index: `idx = floor(point / leaf_size)`
2. Compute a linear hash: `hash = idx_x * Dy * Dz + idx_y * Dz + idx_z`
3. Group points by hash (using `np.unique` + `np.bincount`)
4. Return the centroid (mean x, y, z) of each occupied voxel

This is the NumPy equivalent of PCL's `VoxelGrid` filter.

---

## 5. Octree Plane Fitting (PCA)

**File:** `voxel_map.py` → `VoxelOctoTree.init_plane()`

**Purpose:** Determine if accumulated points in an octree node form a plane,
and if so, compute the plane parameters and their uncertainty.

**Algorithm:**

1. Compute point mean and covariance:
   ```
   center = mean(points)
   C = mean(outer(p, p)) - outer(center, center)
   ```

2. Eigendecomposition of C:
   ```
   eigenvalues: lambda_min <= lambda_mid <= lambda_max
   eigenvectors: e_min (normal), e_mid (y_normal), e_max (x_normal)
   ```

3. **Planarity test:** If `lambda_min < planer_threshold`, the points are
   planar. The plane normal is `e_min`, and `d = -normal . center`.

4. **Plane covariance propagation:** For each point, compute the Jacobian of
   the plane parameters (normal + center) with respect to the point position.
   The key insight: eigenvector perturbation gives
   ```
   dv_min/dp = sum_{m != min} [ e_m @ e_min^T + e_min @ e_m^T ] * (p - center)^T
                                / [n * (lambda_min - lambda_m)]
   ```
   Then `plane_var += J @ point_var @ J^T` for each point.

5. If the planarity test fails, the node subdivides into 8 children (octree split).

---

## 6. State Estimation (Iterative EKF)

**File:** `voxel_map.py` → `VoxelMapManager.state_estimation()`

**Purpose:** Refine the pose estimate by minimizing point-to-plane residuals
against the accumulated voxel map.

**Algorithm (per iteration):**

1. **Transform points** to world frame with current state estimate:
   ```
   p_w = R @ (R_ext @ p_b + T_ext) + T
   ```

2. **Compute per-point covariance** in world frame (sensor model + pose uncertainty):
   ```
   var_w = R_ext @ body_cov @ R_ext^T
         + skew(p_imu) @ rot_cov @ skew(p_imu)^T
         + t_cov
   ```

3. **Find plane correspondences** via voxel lookup:
   - Compute voxel key from world point
   - Traverse octree to find matching plane
   - Compute point-to-plane distance and probability
   - Try neighboring voxel if primary fails

4. **Build Jacobian** for each matched point:
   ```
   H_row = [skew(p_imu) @ R^T @ normal,  normal]   (1 x 6)
   measurement = -distance_to_plane

   R_meas = J_nq @ plane_var @ J_nq^T + normal^T @ var_w @ normal  (scalar)
   ```

5. **Accumulate and solve:**
   ```
   H^T R^{-1} H  += h @ h^T / R_meas         (6 x 6)
   H^T R^{-1} z  += h * measurement / R_meas  (6,)

   K_inv = (H^T R^{-1} H + P^{-1})^{-1}
   delta = K_inv @ H^T R^{-1} z + correction_terms
   state = state (+) delta
   ```

6. **Convergence check:**
   ```
   converged = (||rot_delta|| * 57.3 < 0.01 deg) AND (||pos_delta|| * 100 < 0.015 cm)
   ```

7. Repeat up to `max_iterations` times (default 5). On convergence,
   update the covariance: `P = (I - G) @ P`.

---

## 7. Residual Builder with Plane Cache (JIT)

**File:** `numba_kernels.py` → `build_residual_list_jit()`

**Purpose:** Accelerate the bottleneck of finding plane correspondences for
all points. The standard Python version uses dict lookups and octree traversal;
this version pre-flattens all plane data into contiguous arrays and uses
Numba typed dicts for O(1) lookup.

**Data structures:**
- `plane_normals (P, 3)` — all plane normals concatenated
- `plane_d (P,)` — all plane offsets
- `voxel_plane_starts: Dict[(i64,i64,i64) → i64]` — start index per voxel
- `voxel_plane_counts: Dict[(i64,i64,i64) → i64]` — plane count per voxel

**Algorithm per point:**
1. Compute voxel key from world coordinates
2. Look up plane range `[start, start+count)` in flat arrays
3. For each plane, compute residual math (distance, probability)
4. Keep the plane with highest probability
5. If no match, try neighboring voxel (using voxel center + quarter length)

This eliminates all Python overhead from the inner loop, achieving
near-C++ speed.

---

## 8. World Point Cloud Construction (Phase 2)

**File:** `run.py` → `build_world_pointcloud()`

**Purpose:** After Phase 1 produces odometry, this re-reads the raw bag and
transforms every point into the world frame for the final dense point cloud.

**Algorithm:**
1. Load odometry CSV (timestamp, position, quaternion per scan)
2. For each Livox CustomMsg in the bag:
   a. Parse binary point data (19 bytes per point)
   b. Compute scan-end time from last point's offset
   c. Find nearest odometry pose (within 0.15s tolerance)
   d. Apply blind zone filter
   e. Transform: `p_world = R_odom @ (R_ext @ p_lidar + T_ext) + T_odom`
   f. Accumulate points, intensities, and timestamps
3. Write all points to CSV with timestamps for time-based coloring

Note: This phase does NOT re-run undistortion. The odometry poses were
computed with undistorted points, but the world cloud uses the raw (non-undistorted)
points transformed by the per-scan pose. For most applications this is
sufficient; per-point undistortion would require storing the full IMU trace.
