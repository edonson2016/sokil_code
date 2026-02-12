"""Configuration loader for the LIO pipeline.

Reads YAML config files in the same format as FAST-LIVO2 C++ configs,
mapping the same parameter key paths used in LIVMapper::readParameters().
"""
import yaml
import numpy as np
from dataclasses import dataclass, field


@dataclass
class VoxelMapConfig:
    """Voxel map parameters matching C++ VoxelMapConfig."""
    max_voxel_size: float = 0.5
    max_layer: int = 2
    max_iterations: int = 5
    layer_init_num: list = field(default_factory=lambda: [5, 5, 5, 5, 5])
    max_points_num: int = 50
    planner_threshold: float = 0.01
    beam_err: float = 0.05
    dept_err: float = 0.02
    sigma_num: float = 3.0
    map_sliding_en: bool = False
    half_map_size: int = 100
    sliding_thresh: float = 8.0


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    # Topics
    lid_topic: str = "/livox/lidar"
    imu_topic: str = "/livox/imu"

    # Preprocess
    lidar_type: int = 1  # 1=AVIA, 2=VELO16, 3=OUST64, etc.
    blind: float = 0.8
    point_filter_num: int = 1
    scan_line: int = 6
    filter_size_surf: float = 0.5
    feature_extract_enabled: bool = False

    # IMU
    imu_en: bool = True
    acc_cov: float = 0.5
    gyr_cov: float = 0.3
    imu_int_frame: int = 30
    gravity_est_en: bool = True
    ba_bg_est_en: bool = True
    exposure_estimate_en: bool = False

    # Time offsets
    imu_time_offset: float = 0.0
    lidar_time_offset: float = 0.0

    # Extrinsics: LiDAR-to-IMU
    extrinsic_T: np.ndarray = field(default_factory=lambda: np.zeros(3))
    extrinsic_R: np.ndarray = field(default_factory=lambda: np.eye(3))

    # Voxel map
    voxel: VoxelMapConfig = field(default_factory=VoxelMapConfig)

    # Output
    seq_name: str = "output"
    gravity_align_en: bool = False

    # ROS driver bug fix
    ros_driver_fix_en: bool = False


def load_config(yaml_path: str) -> PipelineConfig:
    """Load configuration from a YAML file.

    Supports the same YAML layout as FAST-LIVO2 config files (e.g., avia.yaml).
    """
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    pc = PipelineConfig()

    # Common
    common = cfg.get('common', {})
    pc.lid_topic = common.get('lid_topic', pc.lid_topic)
    pc.imu_topic = common.get('imu_topic', pc.imu_topic)
    pc.ros_driver_fix_en = common.get('ros_driver_bug_fix', pc.ros_driver_fix_en)

    # Preprocess
    pre = cfg.get('preprocess', {})
    pc.lidar_type = pre.get('lidar_type', pc.lidar_type)
    pc.blind = pre.get('blind', pc.blind)
    pc.point_filter_num = pre.get('point_filter_num', pc.point_filter_num)
    pc.scan_line = pre.get('scan_line', pc.scan_line)
    pc.filter_size_surf = pre.get('filter_size_surf', pc.filter_size_surf)
    pc.feature_extract_enabled = pre.get('feature_extract_enabled', pc.feature_extract_enabled)

    # IMU
    imu = cfg.get('imu', {})
    pc.imu_en = imu.get('imu_en', pc.imu_en)
    pc.acc_cov = imu.get('acc_cov', pc.acc_cov)
    pc.gyr_cov = imu.get('gyr_cov', pc.gyr_cov)
    pc.imu_int_frame = imu.get('imu_int_frame', pc.imu_int_frame)
    pc.gravity_est_en = imu.get('gravity_est_en', pc.gravity_est_en)
    pc.ba_bg_est_en = imu.get('ba_bg_est_en', pc.ba_bg_est_en)

    # Time offsets
    time_off = cfg.get('time_offset', {})
    pc.imu_time_offset = time_off.get('imu_time_offset', pc.imu_time_offset)
    pc.lidar_time_offset = time_off.get('lidar_time_offset', pc.lidar_time_offset)

    # Extrinsics (try 'extrin_calib' first, then 'mapping' for compatibility)
    ext_cfg = cfg.get('extrin_calib', cfg.get('mapping', {}))
    ext_t = ext_cfg.get('extrinsic_T', None)
    if ext_t is not None:
        pc.extrinsic_T = np.array(ext_t, dtype=np.float64).flatten()
    ext_r = ext_cfg.get('extrinsic_R', None)
    if ext_r is not None:
        pc.extrinsic_R = np.array(ext_r, dtype=np.float64).reshape(3, 3)

    # Exposure estimation (under 'vio' key)
    vio = cfg.get('vio', {})
    pc.exposure_estimate_en = vio.get('exposure_estimate_en', pc.exposure_estimate_en)

    # Voxel map (under 'lio' key)
    lio = cfg.get('lio', {})
    vc = pc.voxel
    vc.max_voxel_size = lio.get('voxel_size', vc.max_voxel_size)
    vc.max_layer = lio.get('max_layer', vc.max_layer)
    vc.max_iterations = lio.get('max_iterations', vc.max_iterations)
    vc.planner_threshold = lio.get('min_eigen_value', vc.planner_threshold)
    vc.sigma_num = lio.get('sigma_num', vc.sigma_num)
    vc.dept_err = lio.get('dept_err', vc.dept_err)
    vc.beam_err = lio.get('beam_err', vc.beam_err)
    vc.max_points_num = lio.get('max_points_num', vc.max_points_num)
    layer_init = lio.get('layer_init_num', None)
    if layer_init is not None:
        vc.layer_init_num = list(layer_init)

    # Local map
    local_map = cfg.get('local_map', {})
    vc.map_sliding_en = local_map.get('map_sliding_en', vc.map_sliding_en)
    vc.half_map_size = local_map.get('half_map_size', vc.half_map_size)
    vc.sliding_thresh = local_map.get('sliding_thresh', vc.sliding_thresh)

    # Output
    evo = cfg.get('evo', {})
    pc.seq_name = evo.get('seq_name', pc.seq_name)

    # Gravity alignment
    pc.gravity_align_en = cfg.get('gravity_align_en', pc.gravity_align_en)

    return pc
