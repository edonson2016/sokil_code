#!/usr/bin/env python3
"""FAST-LIVO2 LIO Standalone Pipeline.

One-command processing: rosbag in -> odometry + point clouds + HTML viewer out.

Usage:
    python run.py my_scan.bag
    python run.py my_scan.bag --config custom.yaml
    python run.py my_scan.bag --output-dir results/
    python run.py my_scan.bag --max-viewer-points 1000000

Outputs (all saved to --output-dir, default: same folder as bag):
    1. odometry.csv          - 6-DOF trajectory (timestamp, tx,ty,tz, qx,qy,qz,qw)
    2. raw_scans/            - Per-scan body-frame point clouds (CSV)
    3. world_pointcloud.csv  - All scans transformed to world frame with intensity
    4. viewer.html           - Interactive 3D point cloud viewer (Three.js)
"""
import argparse
import base64
import json
import os
import struct
import sys
import time

import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# Add this directory to path so fast_livo2_lio package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fast_livo2_lio.config import load_config
from fast_livo2_lio.pipeline import LIOPipeline
from fast_livo2_lio.output import write_odometry_csv


# ─────────────────────────────────────────────────────────────────────
# Phase 2: Build world point cloud from odometry + raw bag
# ─────────────────────────────────────────────────────────────────────

def build_world_pointcloud(bag_path, odom_csv_path, config,
                           output_csv_path, blind=0.8):
    """Transform raw LiDAR scans to world frame using odometry poses.

    Args:
        bag_path: Path to the rosbag file.
        odom_csv_path: Path to the odometry CSV output from Phase 1.
        config: PipelineConfig with extrinsic calibration.
        output_csv_path: Where to save the world pointcloud CSV.
        blind: Minimum range filter in meters.

    Returns:
        Tuple of (world_points (N,3), world_intensities (N,),
                  world_timestamps (N,)).
    """
    from rosbags.rosbag1 import Reader

    # Load odometry
    odom = np.genfromtxt(odom_csv_path, delimiter=',', skip_header=1)
    odom_t = odom[:, 0]
    odom_pos = odom[:, 1:4]
    odom_quat = odom[:, 4:8]  # qx, qy, qz, qw
    odom_rots = Rotation.from_quat(odom_quat).as_matrix()  # (N, 3, 3)

    ext_R = config.extrinsic_R
    ext_T = config.extrinsic_T
    lidar_topic = config.lid_topic

    print(f"\n[World Map] Building world point cloud...")
    print(f"[World Map] Using {len(odom_t)} odometry poses")

    # Livox CustomMsg point dtype
    point_dt = np.dtype([
        ('offset_time', '<u4'),
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('reflectivity', 'u1'), ('tag', 'u1'), ('line', 'u1'),
    ])

    all_world_points = []
    all_intensities = []
    all_timestamps = []
    matched = 0
    skipped = 0
    blind_sqr = blind * blind

    with Reader(Path(bag_path)) as reader:
        for conn, ts, rawdata in tqdm(reader.messages(),
                                       total=reader.message_count,
                                       desc="Transforming scans"):
            if conn.topic != lidar_topic:
                continue

            # Check if this is a Livox CustomMsg
            if 'CustomMsg' not in conn.msgtype:
                # For PointCloud2 messages, we'd need a different parser
                # For now, skip non-CustomMsg (could be extended)
                continue

            # Parse Livox CustomMsg header
            data = bytes(rawdata)
            pos = 0
            pos += 4  # seq
            sec = struct.unpack_from('<I', data, pos)[0]; pos += 4
            nsec = struct.unpack_from('<I', data, pos)[0]; pos += 4
            fid_len = struct.unpack_from('<I', data, pos)[0]; pos += 4
            pos += fid_len
            pos += 8  # timebase
            pos += 4  # point_num
            pos += 1  # lidar_id
            pos += 3  # rsvd
            arr_len = struct.unpack_from('<I', data, pos)[0]; pos += 4

            header_time = sec + nsec * 1e-9

            if arr_len == 0:
                continue

            pts = np.frombuffer(data, dtype=point_dt, count=arr_len,
                                offset=pos)

            # Scan end time (matches odometry timestamps)
            last_offset_s = pts['offset_time'][-1] / 1e9
            scan_end_time = header_time + last_offset_s

            # Find closest odometry pose
            idx = np.searchsorted(odom_t, scan_end_time)
            if idx >= len(odom_t):
                idx = len(odom_t) - 1
            elif idx > 0:
                if (abs(odom_t[idx] - scan_end_time) >
                        abs(odom_t[idx - 1] - scan_end_time)):
                    idx = idx - 1

            dt = abs(odom_t[idx] - scan_end_time)
            if dt > 0.15:
                skipped += 1
                continue

            R_odom = odom_rots[idx]
            t_odom = odom_pos[idx]

            # Extract xyz and per-point timestamps
            x = pts['x'].astype(np.float64)
            y = pts['y'].astype(np.float64)
            z = pts['z'].astype(np.float64)
            intensity = pts['reflectivity'].astype(np.float32)
            point_times = header_time + pts['offset_time'].astype(np.float64) / 1e9

            # Filter blind zone
            dist_sq = x ** 2 + y ** 2 + z ** 2
            valid = dist_sq > blind_sqr
            xyz_body = np.column_stack([x[valid], y[valid], z[valid]])
            intensity_valid = intensity[valid]
            times_valid = point_times[valid]

            if len(xyz_body) == 0:
                continue

            # Transform: p_world = R_odom @ (R_ext @ p_lidar + t_ext) + t_odom
            p_imu = (ext_R @ xyz_body.T).T + ext_T
            p_world = (R_odom @ p_imu.T).T + t_odom

            all_world_points.append(p_world)
            all_intensities.append(intensity_valid)
            all_timestamps.append(times_valid)
            matched += 1

    print(f"[World Map] Matched {matched} scans, skipped {skipped}")

    if matched == 0:
        print("[World Map] WARNING: No scans matched to odometry!")
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0)

    world_pts = np.concatenate(all_world_points, axis=0)
    world_int = np.concatenate(all_intensities, axis=0)
    world_ts = np.concatenate(all_timestamps, axis=0)

    print(f"[World Map] Total world points: {len(world_pts):,}")

    # Compute relative timestamps (seconds from start of scan)
    world_ts_rel = (world_ts - world_ts.min()).astype(np.float32)

    # Save CSV
    print(f"[World Map] Saving to {output_csv_path} ...")
    with open(output_csv_path, 'w') as f:
        f.write("x,y,z,intensity,timestamp\n")
        chunk_size = 100000
        for start in tqdm(range(0, len(world_pts), chunk_size),
                          desc="Writing CSV"):
            end = min(start + chunk_size, len(world_pts))
            lines = []
            for i in range(start, end):
                lines.append(f"{world_pts[i, 0]:.4f},{world_pts[i, 1]:.4f},"
                             f"{world_pts[i, 2]:.4f},{world_int[i]:.0f},"
                             f"{world_ts_rel[i]:.4f}\n")
            f.writelines(lines)

    size_mb = os.path.getsize(output_csv_path) / 1e6
    print(f"[World Map] Saved ({size_mb:.1f} MB)")

    return world_pts, world_int, world_ts_rel


# ─────────────────────────────────────────────────────────────────────
# Phase 3: Generate interactive HTML viewer
# ─────────────────────────────────────────────────────────────────────

def generate_html_viewer(world_pts, world_int, output_html_path,
                         max_points=500000, world_ts=None):
    """Generate a self-contained HTML point cloud viewer.

    Args:
        world_pts: (N, 3) world-frame point coordinates.
        world_int: (N,) intensity values.
        output_html_path: Where to save the HTML file.
        max_points: Maximum number of points to embed (for file size).
        world_ts: (N,) per-point timestamps (seconds from scan start).
    """
    N_total = len(world_pts)

    if N_total == 0:
        print("[Viewer] No points to visualize!")
        return

    # If no timestamps provided, create sequential indices as fallback
    if world_ts is None:
        world_ts = np.arange(N_total, dtype=np.float32)

    # Downsample if needed
    if N_total > max_points:
        stride = N_total // max_points
        indices = np.arange(0, N_total, stride)[:max_points]
        pts = world_pts[indices].astype(np.float32)
        ints = world_int[indices].astype(np.float32)
        ts = world_ts[indices].astype(np.float32)
        N = len(pts)
        print(f"[Viewer] Downsampled {N_total:,} -> {N:,} points "
              f"(stride={stride})")
    else:
        pts = world_pts.astype(np.float32)
        ints = world_int.astype(np.float32)
        ts = world_ts.astype(np.float32)
        N = N_total

    # Compute metadata
    center = pts.mean(axis=0).tolist()
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    extent = (maxs - mins).tolist()
    extent = [round(e, 4) for e in extent]
    center = [round(c, 6) for c in center]
    intensity_max = float(ints.max()) if len(ints) > 0 else 1.0
    time_max = float(ts.max()) if len(ts) > 0 else 1.0

    meta = {
        "num_points": N,
        "center": center,
        "extent": extent,
        "intensity_max": intensity_max,
        "time_max": time_max,
    }

    # Pack data: N * (x, y, z, intensity, timestamp) as float32
    packed = np.empty((N, 5), dtype=np.float32)
    packed[:, :3] = pts
    packed[:, 3] = ints
    packed[:, 4] = ts
    b64_data = base64.b64encode(packed.tobytes()).decode('ascii')

    # Generate HTML
    html = _build_viewer_html(meta, b64_data, N_total)

    with open(output_html_path, 'w') as f:
        f.write(html)

    size_mb = os.path.getsize(output_html_path) / 1e6
    print(f"[Viewer] Saved {output_html_path} ({size_mb:.1f} MB, "
          f"{N:,} points)")


def _build_viewer_html(meta, b64_data, total_points):
    """Build the complete HTML string for the 3D viewer."""
    N = meta['num_points']
    cx, cy, cz = meta['center']
    ex, ey, ez = meta['extent']
    max_ext = max(ex, ey, ez)
    time_max_s = meta.get('time_max', 1.0)
    if time_max_s > 60:
        time_label = f"{time_max_s / 60:.1f} min"
    else:
        time_label = f"{time_max_s:.1f} s"

    return f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>World Point Cloud Viewer</title>
<style>
  body {{ margin: 0; overflow: hidden; background: #1a1a2e; font-family: monospace; }}
  #info {{
    position: absolute; top: 10px; left: 10px; color: #eee;
    background: rgba(0,0,0,0.7); padding: 10px 15px; border-radius: 6px;
    font-size: 13px; line-height: 1.6; z-index: 10; pointer-events: none;
  }}
  #info b {{ color: #4fc3f7; }}
  #controls {{
    position: absolute; bottom: 10px; left: 10px; color: #aaa;
    background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 6px;
    font-size: 11px; z-index: 10; pointer-events: none;
  }}
  #colormode {{
    position: absolute; top: 10px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.7); padding: 10px; border-radius: 6px;
  }}
  #colormode button {{
    background: #333; color: #eee; border: 1px solid #555;
    padding: 5px 12px; margin: 2px; border-radius: 4px; cursor: pointer;
    font-family: monospace; font-size: 12px;
  }}
  #colormode button.active {{ background: #4fc3f7; color: #000; border-color: #4fc3f7; }}
  #colormode button:hover {{ background: #555; }}
  #pointsize {{
    position: absolute; top: 60px; right: 10px; z-index: 10;
    background: rgba(0,0,0,0.7); padding: 8px 12px; border-radius: 6px;
    color: #aaa; font-size: 12px; font-family: monospace;
  }}
  #pointsize input {{ width: 100px; vertical-align: middle; }}
</style>
</head>
<body>
<div id="info">
  <b>World Point Cloud</b><br>
  Points: {N:,} (downsampled from {total_points:,})<br>
  Center: ({cx:.1f}, {cy:.1f}, {cz:.1f})<br>
  Extent: {ex:.1f} x {ey:.1f} x {ez:.1f} m<br>
  Scan duration: {time_label}
</div>
<div id="controls">
  Drag: orbit &nbsp;|&nbsp; Scroll: zoom &nbsp;|&nbsp; Right-drag: pan &nbsp;|&nbsp; R: reset view
</div>
<div id="colormode">
  <button class="active" onclick="setColorMode('time', this)">Time</button>
  <button onclick="setColorMode('height', this)">Height</button>
  <button onclick="setColorMode('intensity', this)">Intensity</button>
  <button onclick="setColorMode('distance', this)">Distance</button>
  <button onclick="setColorMode('x', this)">X-axis</button>
  <button onclick="setColorMode('y', this)">Y-axis</button>
</div>
<div id="pointsize">
  Size: <input type="range" min="0.5" max="5" step="0.5" value="1.5"
    oninput="setPointSize(this.value)">
  <span id="psval">1.5</span>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const META = {json.dumps(meta)};
const B64_DATA = "{b64_data}";

const raw = Uint8Array.from(atob(B64_DATA), c => c.charCodeAt(0));
const floats = new Float32Array(raw.buffer);
const N = META.num_points;
const STRIDE = 5;

const positions = new Float32Array(N * 3);
const intensities = new Float32Array(N);
const timestamps = new Float32Array(N);
for (let i = 0; i < N; i++) {{
  positions[i*3]   = floats[i*STRIDE];
  positions[i*3+1] = floats[i*STRIDE+1];
  positions[i*3+2] = floats[i*STRIDE+2];
  intensities[i]   = floats[i*STRIDE+3];
  timestamps[i]    = floats[i*STRIDE+4];
}}

const cx = META.center[0], cy = META.center[1], cz = META.center[2];
const maxExt = Math.max(...META.extent);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1a2e);
const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, maxExt*10);
camera.position.set(cx + maxExt*0.8, cy + maxExt*0.5, cz + maxExt*0.8);
camera.up.set(0, 0, 1);

const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(cx, cy, cz);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.minPolarAngle = 0;
controls.maxPolarAngle = Math.PI;
controls.mouseButtons = {{
  LEFT: THREE.MOUSE.ROTATE,
  MIDDLE: THREE.MOUSE.DOLLY,
  RIGHT: THREE.MOUSE.PAN,
}};
controls.update();

window.addEventListener('keydown', (e) => {{
  if (e.key === 'r' || e.key === 'R') {{
    camera.position.set(cx + maxExt*0.8, cy + maxExt*0.5, cz + maxExt*0.8);
    camera.up.set(0, 0, 1);
    controls.target.set(cx, cy, cz);
    controls.update();
  }}
}});

/* ── Turbo-like colormap (perceptually smooth rainbow) ── */
function turboColormap(t) {{
  /* attempt clamp */
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 0.13572138 + t*(4.6153926 + t*(-42.66032258 + t*(132.13108234 + t*(-152.54895899 + t*58.9161376))))));
  const g = Math.max(0, Math.min(1, 0.09140261 + t*(2.19418839 + t*(4.84296658 + t*(-14.18503333 + t*(4.27729857 + t*2.82956604))))));
  const b = Math.max(0, Math.min(1, 0.1066733 + t*(12.64194608 + t*(-60.58204836 + t*(110.36276771 + t*(-89.90310912 + t*27.34824973))))));
  return [r, g, b];
}}

/* ── Color functions ── */

function timeColor(timestamps, N) {{
  const colors = new Float32Array(N * 3);
  let tmin = Infinity, tmax = -Infinity;
  for (let i = 0; i < N; i++) {{
    if (timestamps[i] < tmin) tmin = timestamps[i];
    if (timestamps[i] > tmax) tmax = timestamps[i];
  }}
  const range = tmax - tmin || 1;
  for (let i = 0; i < N; i++) {{
    const t = (timestamps[i] - tmin) / range;
    const c = turboColormap(t);
    colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
  }}
  return colors;
}}

function heightColor(positions, N) {{
  const colors = new Float32Array(N * 3);
  let zmin = Infinity, zmax = -Infinity;
  for (let i = 0; i < N; i++) {{
    const z = positions[i*3+2];
    if (z < zmin) zmin = z;
    if (z > zmax) zmax = z;
  }}
  const range = zmax - zmin || 1;
  for (let i = 0; i < N; i++) {{
    const t = (positions[i*3+2] - zmin) / range;
    const c = turboColormap(t);
    colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
  }}
  return colors;
}}

function intensityColor(intensities, N) {{
  const colors = new Float32Array(N * 3);
  const imax = META.intensity_max || 1;
  for (let i = 0; i < N; i++) {{
    const t = Math.min(intensities[i] / imax, 1.0);
    /* warm grayscale: dark -> bright amber */
    colors[i*3]   = 0.1 + t * 0.9;
    colors[i*3+1] = 0.08 + t * 0.82;
    colors[i*3+2] = 0.05 + t * 0.55;
  }}
  return colors;
}}

function distanceColor(positions, N) {{
  const colors = new Float32Array(N * 3);
  let dmin = Infinity, dmax = -Infinity;
  for (let i = 0; i < N; i++) {{
    const dx=positions[i*3]-cx, dy=positions[i*3+1]-cy, dz=positions[i*3+2]-cz;
    const d = Math.sqrt(dx*dx+dy*dy+dz*dz);
    if (d<dmin) dmin=d;
    if (d>dmax) dmax=d;
  }}
  const range = dmax - dmin || 1;
  for (let i = 0; i < N; i++) {{
    const dx=positions[i*3]-cx, dy=positions[i*3+1]-cy, dz=positions[i*3+2]-cz;
    const t = (Math.sqrt(dx*dx+dy*dy+dz*dz) - dmin) / range;
    const c = turboColormap(t);
    colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
  }}
  return colors;
}}

function axisColor(positions, N, axis) {{
  const colors = new Float32Array(N * 3);
  let vmin = Infinity, vmax = -Infinity;
  for (let i = 0; i < N; i++) {{
    const v = positions[i*3+axis];
    if (v < vmin) vmin = v;
    if (v > vmax) vmax = v;
  }}
  const range = vmax - vmin || 1;
  for (let i = 0; i < N; i++) {{
    const t = (positions[i*3+axis] - vmin) / range;
    const c = turboColormap(t);
    colors[i*3] = c[0]; colors[i*3+1] = c[1]; colors[i*3+2] = c[2];
  }}
  return colors;
}}

/* ── Scene setup ── */

const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
geometry.setAttribute('color', new THREE.BufferAttribute(timeColor(timestamps, N), 3));

const material = new THREE.PointsMaterial({{ size: 1.5, vertexColors: true, sizeAttenuation: false }});
const points = new THREE.Points(geometry, material);
scene.add(points);

const axesHelper = new THREE.AxesHelper(maxExt * 0.05);
axesHelper.position.set(cx, cy, cz);
scene.add(axesHelper);

window.setColorMode = function(mode, btn) {{
  let colors;
  if (mode === 'time') colors = timeColor(timestamps, N);
  else if (mode === 'height') colors = heightColor(positions, N);
  else if (mode === 'intensity') colors = intensityColor(intensities, N);
  else if (mode === 'distance') colors = distanceColor(positions, N);
  else if (mode === 'x') colors = axisColor(positions, N, 0);
  else if (mode === 'y') colors = axisColor(positions, N, 1);
  else colors = timeColor(timestamps, N);
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  document.querySelectorAll('#colormode button').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
}};

window.setPointSize = function(val) {{
  material.size = parseFloat(val);
  document.getElementById('psval').textContent = val;
}};

window.addEventListener('resize', () => {{
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}});

function animate() {{
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}}
animate();
</script>
</body>
</html>'''


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def find_bag_files(scans_dir):
    """Recursively find all .bag files under a directory."""
    bag_files = []
    for root, dirs, files in os.walk(scans_dir):
        for f in files:
            if f.endswith('.bag'):
                bag_files.append(os.path.join(root, f))
    bag_files.sort()
    return bag_files


def prompt_bag_selection(scans_dir):
    """Scan for .bag files and prompt the user to choose one."""
    bag_files = find_bag_files(scans_dir)

    if not bag_files:
        print(f"Error: No .bag files found in {scans_dir}/")
        sys.exit(1)

    if len(bag_files) == 1:
        print(f"Found 1 bag file: {os.path.relpath(bag_files[0], scans_dir)}")
        answer = input("Process this bag? [Y/n]: ").strip().lower()
        if answer in ('', 'y', 'yes'):
            return bag_files[0]
        else:
            print("Aborted.")
            sys.exit(0)

    print(f"\nFound {len(bag_files)} bag file(s) in {scans_dir}/:\n")
    for i, bf in enumerate(bag_files, 1):
        rel = os.path.relpath(bf, scans_dir)
        size_mb = os.path.getsize(bf) / 1e6
        print(f"  [{i}] {rel}  ({size_mb:.1f} MB)")

    print()
    while True:
        choice = input(f"Which bag would you like to expand into a pointcloud? [1-{len(bag_files)}]: ").strip()
        try:
            idx = int(choice)
            if 1 <= idx <= len(bag_files):
                return bag_files[idx - 1]
        except ValueError:
            pass
        print(f"Please enter a number between 1 and {len(bag_files)}.")


def main():
    parser = argparse.ArgumentParser(
        description='FAST-LIVO2 LIO Standalone Pipeline\n\n'
                    'Process a rosbag and produce:\n'
                    '  1. Odometry CSV\n'
                    '  2. Raw per-scan point clouds\n'
                    '  3. World-frame point cloud\n'
                    '  4. Interactive 3D HTML viewer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('bag', nargs='?', default=None,
                        help='Path to ROS1 .bag file '
                             '(if omitted, scans Scans/ folder)')
    parser.add_argument('--config', default=None,
                        help='Path to YAML config file '
                             '(default: avia.yaml in this folder)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory '
                             '(default: same folder as bag file)')
    parser.add_argument('--lid-topic', default=None,
                        help='Override LiDAR topic from config')
    parser.add_argument('--imu-topic', default=None,
                        help='Override IMU topic from config')
    parser.add_argument('--max-viewer-points', type=int, default=500000,
                        help='Max points in HTML viewer '
                             '(default: 500000)')
    parser.add_argument('--skip-raw-scans', action='store_true',
                        help='Skip saving per-scan raw point clouds')

    args = parser.parse_args()

    # Resolve bag path: use argument if provided, otherwise scan Scans/ folder
    if args.bag:
        bag_path = os.path.abspath(args.bag)
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        scans_dir = os.path.join(script_dir, 'Scans')
        if not os.path.isdir(scans_dir):
            print(f"Error: Scans/ folder not found at {scans_dir}")
            print("Either provide a bag file path or place bags in the Scans/ folder.")
            sys.exit(1)
        bag_path = prompt_bag_selection(scans_dir)

    if not os.path.isfile(bag_path):
        print(f"Error: Bag file not found: {bag_path}")
        sys.exit(1)

    # Config
    if args.config:
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'avia.yaml')

    if not os.path.isfile(config_path):
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    # Output directory
    if args.output_dir:
        out_dir = os.path.abspath(args.output_dir)
    else:
        out_dir = os.path.dirname(bag_path)
    os.makedirs(out_dir, exist_ok=True)

    # Output paths
    odom_path = os.path.join(out_dir, 'odometry.csv')
    raw_scan_dir = os.path.join(out_dir, 'raw_scans') if not args.skip_raw_scans else None
    world_csv_path = os.path.join(out_dir, 'world_pointcloud.csv')
    viewer_path = os.path.join(out_dir, 'viewer.html')

    print("=" * 60)
    print("  FAST-LIVO2 LIO Standalone Pipeline")
    print("=" * 60)
    print(f"  Bag:        {bag_path}")
    print(f"  Config:     {config_path}")
    print(f"  Output dir: {out_dir}")
    print(f"  Outputs:")
    print(f"    1. Odometry:          {odom_path}")
    if raw_scan_dir:
        print(f"    2. Raw scans:         {raw_scan_dir}/")
    else:
        print(f"    2. Raw scans:         (skipped)")
    print(f"    3. World pointcloud:  {world_csv_path}")
    print(f"    4. HTML viewer:       {viewer_path}")
    print("=" * 60)

    t0 = time.time()

    # ── Phase 1: Run LIO pipeline ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 1: LiDAR-Inertial Odometry")
    print("=" * 60)

    config = load_config(config_path)
    if args.lid_topic:
        config.lid_topic = args.lid_topic
    if args.imu_topic:
        config.imu_topic = args.imu_topic

    pipeline = LIOPipeline(config)
    pipeline.run(bag_path, odom_path, scan_output_dir=raw_scan_dir)

    t1 = time.time()
    print(f"\n[Phase 1] Complete in {t1 - t0:.1f}s")

    # ── Phase 2: Build world point cloud ─────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 2: World Point Cloud")
    print("=" * 60)

    world_pts, world_int, world_ts = build_world_pointcloud(
        bag_path, odom_path, config, world_csv_path,
        blind=config.blind,
    )

    t2 = time.time()
    print(f"[Phase 2] Complete in {t2 - t1:.1f}s")

    # ── Phase 3: Generate HTML viewer ────────────────────────────────
    print("\n" + "=" * 60)
    print("  Phase 3: Interactive 3D Viewer")
    print("=" * 60)

    if len(world_pts) > 0:
        generate_html_viewer(
            world_pts, world_int, viewer_path,
            max_points=args.max_viewer_points,
            world_ts=world_ts,
        )
    else:
        print("[Viewer] No points available, skipping viewer generation.")

    t3 = time.time()
    print(f"[Phase 3] Complete in {t3 - t2:.1f}s")

    # ── Summary ──────────────────────────────────────────────────────
    total = t3 - t0
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"  Total time: {total:.1f}s ({total / 60:.1f} min)")
    print(f"\n  Outputs:")
    print(f"    1. {odom_path}")
    if raw_scan_dir:
        n_scans = len([f for f in os.listdir(raw_scan_dir)
                       if f.endswith('.csv')]) if os.path.isdir(raw_scan_dir) else 0
        print(f"    2. {raw_scan_dir}/ ({n_scans} scans)")
    print(f"    3. {world_csv_path} "
          f"({os.path.getsize(world_csv_path) / 1e6:.1f} MB)"
          if os.path.isfile(world_csv_path) else "")
    if os.path.isfile(viewer_path):
        print(f"    4. {viewer_path} "
              f"({os.path.getsize(viewer_path) / 1e6:.1f} MB)")
    print()


if __name__ == '__main__':
    main()
