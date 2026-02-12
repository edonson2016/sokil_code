"""ROS1 bag file reader using the rosbags library (no ROS install needed).

Parses sensor_msgs/PointCloud2, sensor_msgs/Imu, and
livox_ros_driver/CustomMsg messages from .bag files.
"""
import struct
import numpy as np
from pathlib import Path

from rosbags.rosbag1 import Reader
from rosbags.typesys import Stores, get_typestore
from tqdm import tqdm

from .types import ImuData, LidarScan


# Standard field type sizes from PointCloud2 spec
_POINTFIELD_DTYPES = {
    1: ('B', 1),   # UINT8
    2: ('b', 1),   # INT8
    3: ('H', 2),   # UINT16
    4: ('h', 2),   # INT16
    5: ('I', 4),   # UINT32
    6: ('i', 4),   # INT32
    7: ('f', 4),   # FLOAT32
    8: ('d', 8),   # FLOAT64
}

# Numpy dtype mapping
_POINTFIELD_NP_DTYPES = {
    1: np.uint8,
    2: np.int8,
    3: np.uint16,
    4: np.int16,
    5: np.uint32,
    6: np.int32,
    7: np.float32,
    8: np.float64,
}


def _find_field(fields, name):
    """Find a field by name in PointCloud2 fields list."""
    for f in fields:
        if f.name == name:
            return f
    return None


def _find_time_field(fields):
    """Auto-detect the per-point time field.

    Different LiDARs use different field names:
    - Velodyne: 'time' (float32, seconds)
    - Ouster: 't' (uint32, nanoseconds)
    - Hesai/Pandar: 'timestamp' (float64, absolute seconds)
    - Robosense: 'timestamp' (float64, absolute seconds)
    - Livox: 'offset_time' (uint32, nanoseconds)

    Returns: (field, field_type_str) or (None, None)
    """
    candidates = ['time', 't', 'timestamp', 'offset_time']
    for name in candidates:
        f = _find_field(fields, name)
        if f is not None:
            return f, name
    return None, None


def _parse_pointcloud2(msg, typestore):
    """Parse a PointCloud2 message into numpy arrays.

    Returns:
        xyz: (N, 3) float64
        intensities: (N,) float32
        time_offsets_ms: (N,) float64 - per-point time offset from scan start in ms
    """
    fields = msg.fields
    point_step = msg.point_step
    width = msg.width
    height = msg.height
    data = bytes(msg.data)
    n_points = width * height

    if n_points == 0:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

    # Find x, y, z fields
    fx = _find_field(fields, 'x')
    fy = _find_field(fields, 'y')
    fz = _find_field(fields, 'z')
    fi = _find_field(fields, 'intensity')

    if fx is None or fy is None or fz is None:
        raise ValueError("PointCloud2 missing x/y/z fields")

    # Build structured dtype for efficient parsing
    buf = np.frombuffer(data, dtype=np.uint8).reshape(n_points, point_step)

    # Extract xyz
    xyz = np.zeros((n_points, 3), dtype=np.float64)
    for dim, ff in enumerate([fx, fy, fz]):
        dt = _POINTFIELD_NP_DTYPES[ff.datatype]
        sz = _POINTFIELD_DTYPES[ff.datatype][1]
        raw = buf[:, ff.offset:ff.offset + sz].copy()
        xyz[:, dim] = raw.view(dt).flatten().astype(np.float64)

    # Extract intensity
    if fi is not None:
        dt = _POINTFIELD_NP_DTYPES[fi.datatype]
        sz = _POINTFIELD_DTYPES[fi.datatype][1]
        raw = buf[:, fi.offset:fi.offset + sz].copy()
        intensities = raw.view(dt).flatten().astype(np.float32)
    else:
        intensities = np.zeros(n_points, dtype=np.float32)

    # Extract time offset
    ft, ft_name = _find_time_field(fields)
    if ft is not None:
        dt = _POINTFIELD_NP_DTYPES[ft.datatype]
        sz = _POINTFIELD_DTYPES[ft.datatype][1]
        raw = buf[:, ft.offset:ft.offset + sz].copy()
        time_raw = raw.view(dt).flatten().astype(np.float64)

        # Normalize to ms offset from first point
        if ft_name == 't' or ft_name == 'offset_time':
            # Nanoseconds -> ms
            time_offsets_ms = time_raw / 1e6
        elif ft_name == 'timestamp':
            # Absolute seconds -> relative ms
            time_offsets_ms = (time_raw - time_raw[0]) * 1000.0
        elif ft_name == 'time':
            # Seconds -> ms (already relative for Velodyne)
            time_offsets_ms = time_raw * 1000.0
        else:
            time_offsets_ms = time_raw * 1000.0

        # Ensure non-negative and relative to start
        if len(time_offsets_ms) > 0 and time_offsets_ms[0] != 0:
            time_offsets_ms = time_offsets_ms - time_offsets_ms[0]
    else:
        # No time field: assume uniform distribution over scan
        time_offsets_ms = np.linspace(0, 100.0, n_points)

    return xyz, intensities, time_offsets_ms


def _parse_livox_custommsg(rawdata):
    """Parse a livox_ros_driver/CustomMsg from raw bytes.

    CustomMsg layout (ROS1 serialized):
      Header header        (uint32 seq, uint32 sec, uint32 nsec, string frame_id)
      uint64 timebase      (ns since epoch)
      uint32 point_num
      uint8  lidar_id
      uint8[3] rsvd
      CustomPoint[] points (uint32 array_len prefix, then N * 19 bytes)

    CustomPoint: uint32 offset_time(ns), float32 x, float32 y, float32 z,
                 uint8 reflectivity, uint8 tag, uint8 line

    Returns:
        header_time: float64 seconds since epoch
        xyz: (N, 3) float64
        intensities: (N,) float32
        time_offsets_ms: (N,) float64
    """
    data = bytes(rawdata)
    pos = 0

    # Header
    _seq = struct.unpack_from('<I', data, pos)[0]; pos += 4
    sec = struct.unpack_from('<I', data, pos)[0]; pos += 4
    nsec = struct.unpack_from('<I', data, pos)[0]; pos += 4
    fid_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
    pos += fid_len  # skip frame_id string

    # CustomMsg fields
    _timebase = struct.unpack_from('<Q', data, pos)[0]; pos += 8
    _point_num = struct.unpack_from('<I', data, pos)[0]; pos += 4
    _lidar_id = struct.unpack_from('<B', data, pos)[0]; pos += 1
    pos += 3  # rsvd

    # Points array
    arr_len = struct.unpack_from('<I', data, pos)[0]; pos += 4

    header_time = sec + nsec * 1e-9

    if arr_len == 0:
        return header_time, np.zeros((0, 3)), np.zeros(0), np.zeros(0)

    # Parse all points at once using structured numpy dtype
    # CustomPoint: uint32, float32, float32, float32, uint8, uint8, uint8 = 19 bytes
    point_dt = np.dtype([
        ('offset_time', '<u4'),
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('reflectivity', 'u1'), ('tag', 'u1'), ('line', 'u1'),
    ])
    points_buf = np.frombuffer(data, dtype=point_dt, count=arr_len, offset=pos)

    xyz = np.column_stack([
        points_buf['x'].astype(np.float64),
        points_buf['y'].astype(np.float64),
        points_buf['z'].astype(np.float64),
    ])
    intensities = points_buf['reflectivity'].astype(np.float32)
    time_offsets_ms = points_buf['offset_time'].astype(np.float64) / 1e6  # ns -> ms

    return header_time, xyz, intensities, time_offsets_ms


def _parse_imu(msg) -> ImuData:
    """Parse a sensor_msgs/Imu message into ImuData."""
    stamp = msg.header.stamp
    ts = stamp.sec + stamp.nanosec * 1e-9

    acc = np.array([
        msg.linear_acceleration.x,
        msg.linear_acceleration.y,
        msg.linear_acceleration.z
    ])
    gyr = np.array([
        msg.angular_velocity.x,
        msg.angular_velocity.y,
        msg.angular_velocity.z
    ])
    return ImuData(timestamp=ts, acc=acc, gyr=gyr)


def read_bag(bag_path: str, lidar_topic: str, imu_topic: str,
             lidar_time_offset: float = 0.0,
             imu_time_offset: float = 0.0):
    """Read a ROS1 bag file and yield sensor data chronologically.

    Args:
        bag_path: Path to the .bag file.
        lidar_topic: Topic name for PointCloud2 messages.
        imu_topic: Topic name for IMU messages.
        lidar_time_offset: Time offset to add to LiDAR timestamps.
        imu_time_offset: Time offset to subtract from IMU timestamps.

    Yields:
        Tuples of ('lidar', LidarScan) or ('imu', ImuData), sorted by timestamp.
    """
    typestore = get_typestore(Stores.ROS1_NOETIC)
    bag = Path(bag_path)

    lidar_msgs = []
    imu_msgs = []

    with Reader(bag) as reader:
        msg_count = reader.message_count
        for connection, timestamp, rawdata in tqdm(
            reader.messages(), total=msg_count,
            desc="Reading bag", unit="msg", dynamic_ncols=True,
        ):
            topic = connection.topic
            if topic == lidar_topic:
                if 'CustomMsg' in connection.msgtype:
                    # Livox CustomMsg: parse raw bytes directly
                    header_time, xyz, intensities, times_ms = (
                        _parse_livox_custommsg(rawdata))
                    header_time += lidar_time_offset
                else:
                    # Standard PointCloud2
                    msg = typestore.deserialize_ros1(
                        rawdata, connection.msgtype)
                    xyz, intensities, times_ms = _parse_pointcloud2(
                        msg, typestore)
                    stamp = msg.header.stamp
                    header_time = (stamp.sec + stamp.nanosec * 1e-9
                                   + lidar_time_offset)

                if len(xyz) > 1:
                    scan = LidarScan(
                        header_time=header_time,
                        points=xyz,
                        intensities=intensities,
                        timestamps_ms=times_ms,
                    )
                    lidar_msgs.append(('lidar', header_time, scan))

            elif topic == imu_topic:
                msg = typestore.deserialize_ros1(rawdata, connection.msgtype)
                imu = _parse_imu(msg)
                imu.timestamp -= imu_time_offset
                imu_msgs.append(('imu', imu.timestamp, imu))

    # Merge and sort by timestamp
    all_msgs = lidar_msgs + imu_msgs
    all_msgs.sort(key=lambda x: x[1])

    for msg_type, _, data in all_msgs:
        yield msg_type, data
