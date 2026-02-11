# Data Structures & Coordinate Frames

## Coordinate Frames

```
LiDAR Body Frame (b)        IMU/Body Frame (i)           World Frame (w)
  ┌───────────┐    R_ext, T_ext    ┌───────────┐   R_odom, T_odom   ┌───────────┐
  │  Raw scan │ ────────────────►  │  IMU body │ ────────────────►  │   World   │
  │  points   │                    │   frame   │                    │   frame   │
  └───────────┘                    └───────────┘                    └───────────┘

Transforms:
  p_imu   = R_ext @ p_lidar + T_ext        (LiDAR → IMU extrinsic calibration)
  p_world = R_odom @ p_imu + T_odom        (IMU → world via odometry pose)
  p_world = R_odom @ (R_ext @ p_lidar + T_ext) + T_odom   (combined)
```

**R_ext, T_ext** are fixed calibration parameters from `avia.yaml`:
```yaml
extrin_calib:
  extrinsic_T: [0.04165, 0.02326, -0.0284]   # meters
  extrinsic_R: [1, 0, 0, 0, 1, 0, 0, 0, 1]   # 3x3 row-major
```

**R_odom, T_odom** are estimated per scan by the EKF in Phase 1.

---

## Core Data Structures

### LidarScan (types.py)

Represents one complete LiDAR sweep after parsing from the bag.

```python
@dataclass
class LidarScan:
    header_time: float          # Scan start time (seconds since epoch)
    points: np.ndarray          # (N, 3) xyz in LiDAR body frame
    intensities: np.ndarray     # (N,) reflectivity values [0-255]
    timestamps_ms: np.ndarray   # (N,) per-point time offset from header_time in ms
```

The `timestamps_ms` field is critical for motion undistortion — it records when
each point was actually measured during the sweep. Points are sorted by this
time after preprocessing.

### ImuData (types.py)

A single IMU measurement.

```python
@dataclass
class ImuData:
    timestamp: float            # Absolute time (seconds since epoch)
    acc: np.ndarray             # (3,) linear acceleration in m/s^2
    gyr: np.ndarray             # (3,) angular velocity in rad/s
```

### StatesGroup (state.py)

The 19-dimensional EKF state. Updated every scan.

```python
class StatesGroup:
    rot_end: np.ndarray         # (3, 3) rotation matrix (world ← body)
    pos_end: np.ndarray         # (3,) position in world frame
    vel_end: np.ndarray         # (3,) velocity in world frame
    inv_expo_time: float        # Inverse exposure time
    bias_g: np.ndarray          # (3,) gyroscope bias
    bias_a: np.ndarray          # (3,) accelerometer bias
    gravity: np.ndarray         # (3,) gravity vector in world frame
    cov: np.ndarray             # (19, 19) error-state covariance matrix
```

**Boxplus semantics** (`state (+) delta`):
- Rotation uses SO(3) exponential map: `R_new = R_old @ Exp(delta[0:3])`
- All other components use standard addition

**State vector layout** for boxplus/boxminus delta:
```
Index    Component           Description
[0:3]    rotation            SO(3) Lie algebra element
[3:6]    position            xyz delta
[6]      inv_expo_time       scalar delta
[7:10]   velocity            xyz delta
[10:13]  bias_g              gyro bias delta
[13:16]  bias_a              accel bias delta
[16:19]  gravity             gravity vector delta
```

### Pose6D (types.py)

Snapshot of IMU state at one instant during a scan. Used for undistortion.

```python
@dataclass
class Pose6D:
    offset_time: float          # Time offset from scan start (seconds)
    acc: np.ndarray             # (3,) acceleration at this instant
    gyr: np.ndarray             # (3,) angular velocity at this instant
    vel: np.ndarray             # (3,) velocity at this instant
    pos: np.ndarray             # (3,) position at this instant
    rot: np.ndarray             # (3, 3) rotation at this instant
```

During undistortion, LiDAR points are matched to IMU bracket intervals
defined by consecutive `Pose6D` entries, then each point is back-projected
to the scan-end frame using interpolated rotation and position.

### VoxelOctoTree (voxel_map.py)

Octree node within a single voxel cell. Each voxel can subdivide up to
`max_layer` levels to achieve finer plane fitting.

```python
class VoxelOctoTree:
    plane_ptr: VoxelPlane       # Fitted plane (if leaf is planar)
    leaves: list[8]             # 8 children (octree subdivision)
    layer: int                  # Current depth (0 = root)
    voxel_center: np.ndarray    # (3,) center of this octree node
    quarter_length: float       # Half the node's half-edge (for subdivision)
    temp_points: list           # Accumulated (point, covariance) pairs
    octo_state: int             # 0 = leaf, 1 = branch (subdivided)
```

**Plane fitting:** When enough points accumulate (`points_size_threshold`),
PCA eigendecomposition is run on the point covariance matrix. If the smallest
eigenvalue is below `planer_threshold`, the points form a good plane and the
node becomes a leaf. Otherwise, it subdivides into 8 children.

### VoxelPlane (voxel_map.py)

Fitted plane within an octree leaf.

```python
class VoxelPlane:
    center: np.ndarray          # (3,) mean of points on the plane
    normal: np.ndarray          # (3,) plane normal (smallest eigenvector)
    covariance: np.ndarray      # (3, 3) point distribution covariance
    plane_var: np.ndarray       # (6, 6) plane parameter uncertainty
    d: float                    # Plane offset: normal . center + d = 0
    radius: float               # sqrt(largest eigenvalue) — plane extent
    min_eigen: float            # Smallest eigenvalue (flatness measure)
    is_plane: bool              # True if eigenvalue test passed
```

The `plane_var` (6x6) encodes uncertainty in both the plane normal (first 3)
and center (last 3), propagated from individual point covariances via Jacobians.

---

## Sensor Covariance Model

Body-frame covariance for each LiDAR point (see `calc_body_cov` in `voxel_map.py`):

```
cov = outer(direction, direction) * range_variance
    + r^2 * skew(direction) @ N @ diag(angular_var) @ N^T @ skew(direction)^T
```

Where:
- `direction` = unit vector from origin to point
- `range_variance` = `dept_err^2` (depth measurement noise)
- `angular_var` = `sin(beam_err)^2` (beam divergence)
- `N` = orthonormal tangent basis perpendicular to direction
- `r` = distance to point

This models range error along the beam direction and angular error
perpendicular to it.

---

## Voxel Map Hash Table

The voxel map is a Python dict mapping `(i, j, k)` integer keys to octree roots:

```python
voxel_map: dict[tuple[int, int, int], VoxelOctoTree]
```

Key computation (matching C++ integer truncation):
```python
loc = point_world / voxel_size
key = (int(loc[0] - 1) if loc[0] < 0 else int(loc[0]),
       int(loc[1] - 1) if loc[1] < 0 else int(loc[1]),
       int(loc[2] - 1) if loc[2] < 0 else int(loc[2]))
```

---

## Livox CustomMsg Binary Layout

The Livox Avia uses a custom ROS message type. Binary format in the bag:

```
Header:
  uint32  seq
  uint32  sec           ← timestamp seconds
  uint32  nsec          ← timestamp nanoseconds
  uint32  frame_id_len
  char[]  frame_id      ← variable length string
  uint64  timebase      ← nanoseconds since epoch
  uint32  point_num
  uint8   lidar_id
  uint8[3] reserved
  uint32  array_len     ← number of points

Points (19 bytes each):
  uint32  offset_time   ← nanoseconds from timebase
  float32 x
  float32 y
  float32 z
  uint8   reflectivity  ← intensity [0-255]
  uint8   tag           ← point classification
  uint8   line          ← scan line number
```
