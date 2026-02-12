#!/usr/bin/env python3
"""
Ground Plane Analysis for LiDAR Pointclouds
============================================

This script analyses a LiDAR pointcloud CSV file by:

  1. **Loading** a CSV pointcloud (columns: x, y, z, intensity, timestamp).

  2. **Ground plane detection** via RANSAC with an SVD refinement step.
     - Randomly samples 3-point subsets to propose candidate planes.
     - Rejects any plane whose normal is tilted more than MAX_TILT_DEG
       from vertical [0, 0, 1] (default 20 degrees).
     - The plane with the most inliers (points within RANSAC_THRESHOLD
       of the plane, default 2 cm) wins.
     - The winning plane is refined by SVD on its inliers for sub-sample
       accuracy.
     - RANSAC always runs on the **entire** pointcloud (no bounding-box
       restriction), so the ground plane estimate is globally optimal.

  3. **Rotation** of the full pointcloud so the detected ground plane
     aligns with Z = 0.  Uses Rodrigues' rotation via
     ``scipy.spatial.transform.Rotation``.  The ground-plane inliers'
     mean Z is shifted to exactly 0.

  4. **Auto-crop bounds** computed via percentile trimming (default 95 %)
     to find the dense scan region, discarding sparse outliers.

  5. **Interactive 3D HTML viewer** (Plotly) of the rotated, cropped
     pointcloud, saved alongside the input CSV.

  6. **Rectangular heatmap slices** at configurable Z intervals (default
     2 cm thick, from -0.50 m to +1.00 m above ground).
     - Each slice is binned on a uniform rectangular grid whose cell
       size is configurable (default 1.5 cm).
     - Raw bin counts are multiplied by r^2 (distance from the scanner
       origin) to compensate for the natural inverse-square falloff of
       LiDAR point density.
     - Plots are oriented landscape (Y horizontal, X vertical) and use
       a fixed subplot layout so the image does not shift between frames.

  7. **Click-through HTML viewer** embedding all slice images as base64
     data URIs for instant, jitter-free comparison.
     - Prev / Next buttons, a range-slider scrubber bar, keyboard
       navigation (arrow keys, Space, Home, End).

  8. **Summary report** (ground_plane_report.txt) with all parameters,
     plane equation, rotation info, and per-slice point counts.

Usage
-----
Run interactively — the script will prompt for every parameter it needs::

    python3 ground_plane_analysis.py

All output files (heatmaps/, viewer HTMLs, report) are saved **in the
same directory as the input CSV**, so results stay next to the data they
came from.

Dependencies
------------
- numpy, pandas, matplotlib, scipy, plotly

Author
------
Generated with Claude Code.
"""

import os
import sys
import time
import json
import http.server
import threading
import webbrowser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import plotly.graph_objects as go


# ============================================================
# DEFAULTS  (used when the user presses Enter at a prompt)
# ============================================================
DEFAULT_RANSAC_ITERATIONS = 1000
DEFAULT_RANSAC_THRESHOLD = 0.02        # metres
DEFAULT_RANSAC_SUBSAMPLE = 1_000_000
DEFAULT_MAX_TILT_DEG = 20.0

DEFAULT_SLICE_THICKNESS = 0.02         # metres
DEFAULT_SLICE_MIN = -0.50              # metres below ground
DEFAULT_SLICE_MAX = 1.00               # metres above ground
DEFAULT_BIN_SIZE = 0.015               # 1.5 cm

DEFAULT_AUTO_CROP_PERCENTILE = 95.0
DEFAULT_DPI = 150


# ============================================================
# INTERACTIVE PROMPTS
# ============================================================

def ask(prompt, default, cast=float):
    """
    Prompt the user for a value with a default.

    Parameters
    ----------
    prompt : str
        Question text shown to the user.
    default : any
        Value used when the user presses Enter without typing.
    cast : callable
        Type to convert the input string to (float, int, str …).

    Returns
    -------
    The user's answer, converted to ``cast`` type.
    """
    raw = input(f"  {prompt} [{default}]: ").strip()
    if raw == "":
        return cast(default)
    return cast(raw)


def ask_optional(prompt, default, cast=float):
    """
    Like ``ask()``, but the user may type ``none`` to get ``None``.

    Useful for optional bounding-box limits.  Pass ``default="none"``
    (or ``default=None``) to default to unbounded.
    """
    display = default if default is not None else "none"
    raw = input(f"  {prompt} [{display}]: ").strip()
    if raw == "":
        if default is None or str(default).lower() == "none":
            return None
        return cast(default)
    if raw.lower() == "none":
        return None
    return cast(raw)


def find_pointclouds(base_dir):
    """
    Recursively search *base_dir* for CSV files that look like pointclouds.

    A file qualifies if its first line contains the header columns
    ``x``, ``y``, ``z``.  Returns a sorted list of absolute paths.
    """
    candidates = []
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if not f.lower().endswith(".csv"):
                continue
            path = os.path.join(root, f)
            try:
                with open(path, "r") as fh:
                    header = fh.readline().lower()
                if "x" in header and "y" in header and "z" in header:
                    candidates.append(path)
            except Exception:
                pass
    return sorted(candidates)


def prompt_for_config():
    """
    Walk the user through every configurable parameter, returning a dict.

    The dict keys mirror the old module-level constants so the rest of
    the code can read them unchanged.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)

    # ---- Input CSV ----
    print("\n=== Ground Plane Analysis ===\n")
    print("Looking for pointcloud CSVs ...")
    csvs = find_pointclouds(base_dir)

    if csvs:
        print(f"  Found {len(csvs)} candidate file(s):\n")
        for i, p in enumerate(csvs):
            # Show path relative to base_dir for readability
            rel = os.path.relpath(p, base_dir)
            print(f"    [{i + 1}] {rel}")
        print()
        choice = input("  Enter number to select, or full path to a CSV: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(csvs):
            input_csv = csvs[int(choice) - 1]
        else:
            input_csv = choice
    else:
        input_csv = input("  No CSVs found.  Enter full path to pointcloud CSV: ").strip()

    if not os.path.isfile(input_csv):
        print(f"  ERROR: file not found: {input_csv}")
        sys.exit(1)

    print(f"\n  Selected: {input_csv}\n")

    # Output goes next to the input CSV
    output_dir = os.path.dirname(input_csv)
    heatmap_dir = os.path.join(output_dir, "heatmaps")

    # ---- RANSAC parameters ----
    print("--- RANSAC Ground-Plane Detection ---")
    ransac_iter = ask("RANSAC iterations", DEFAULT_RANSAC_ITERATIONS, int)
    ransac_thresh = ask("Inlier threshold (m)", DEFAULT_RANSAC_THRESHOLD)
    ransac_sub = ask("Subsample size for RANSAC voting", DEFAULT_RANSAC_SUBSAMPLE, int)
    max_tilt = ask("Max tilt from vertical (deg)", DEFAULT_MAX_TILT_DEG)

    # ---- Auto-crop ----
    print("\n--- Auto-Crop (percentile trimming) ---")
    crop_pct = ask("Keep percentage of points per axis", DEFAULT_AUTO_CROP_PERCENTILE)

    dpi = ask("PNG DPI", DEFAULT_DPI, int)

    print("\n  Heatmap bounds, slice range, and bin size will be set")
    print("  in the interactive 3D viewer (opens in your browser).\n")

    return {
        "INPUT_CSV": input_csv,
        "OUTPUT_DIR": output_dir,
        "HEATMAP_DIR": heatmap_dir,
        "RANSAC_ITERATIONS": ransac_iter,
        "RANSAC_THRESHOLD": ransac_thresh,
        "RANSAC_SUBSAMPLE": ransac_sub,
        "MAX_TILT_DEG": max_tilt,
        "AUTO_CROP_PERCENTILE": crop_pct,
        "DPI": dpi,
        # These will be filled by the interactive HTML viewer:
        "SLICE_THICKNESS": None,
        "SLICE_MIN": None,
        "SLICE_MAX": None,
        "BIN_SIZE": None,
        "SLICE_X_MIN": None,
        "SLICE_X_MAX": None,
        "SLICE_Y_MIN": None,
        "SLICE_Y_MAX": None,
    }


# ============================================================
# CORE FUNCTIONS
# ============================================================

def load_pointcloud(path):
    """
    Load a CSV pointcloud into numpy arrays.

    Expected columns: x, y, z, intensity, timestamp.
    All values are read as float32 to save memory on large clouds.

    Parameters
    ----------
    path : str
        Absolute path to the CSV file.

    Returns
    -------
    xyz : ndarray, shape (N, 3), dtype float32
        Point positions.
    intensity : ndarray, shape (N,), dtype float32
        Per-point intensity values.
    timestamp : ndarray, shape (N,), dtype float32
        Per-point timestamps.
    """
    print(f"  Reading {path} ...")
    df = pd.read_csv(path, dtype={
        'x': np.float32, 'y': np.float32, 'z': np.float32,
        'intensity': np.float32, 'timestamp': np.float32,
    })
    xyz = df[['x', 'y', 'z']].values
    intensity = df['intensity'].values
    timestamp = df['timestamp'].values
    print(f"  Loaded {len(xyz):,} points")
    return xyz, intensity, timestamp


def ransac_plane_fit(points, n_iter, threshold, subsample_size, max_tilt_deg):
    """
    RANSAC plane detection with a tilt constraint.

    Finds the plane  ``n . p + d = 0``  with the most inliers (points
    closer than *threshold*), among planes whose normal is within
    *max_tilt_deg* of vertical [0, 0, 1].

    The algorithm subsamples *subsample_size* points for the voting
    phase (speed), then evaluates the winning plane on the full cloud
    to produce the final inlier mask.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    n_iter : int
        Number of RANSAC iterations.
    threshold : float
        Inlier distance threshold (metres).
    subsample_size : int
        Points used during the voting loop.
    max_tilt_deg : float
        Maximum acceptable tilt from vertical (degrees).

    Returns
    -------
    best_normal : ndarray, shape (3,)
        Unit normal of the best plane (pointing upward).
    best_d : float
        Offset term of the plane equation.
    inlier_mask : ndarray, shape (N,), dtype bool
        True for points within *threshold* of the best plane.
    """
    rng = np.random.default_rng(42)
    n_points = len(points)
    cos_max_tilt = np.cos(np.radians(max_tilt_deg))

    # Subsample for speed
    if n_points > subsample_size:
        sub_idx = rng.choice(n_points, size=subsample_size, replace=False)
        subsample = points[sub_idx]
    else:
        subsample = points

    best_inlier_count = 0
    best_normal = None
    best_d = None

    for i in range(n_iter):
        # Pick 3 random points and compute a candidate plane
        idx = rng.choice(len(subsample), size=3, replace=False)
        p1, p2, p3 = subsample[idx]

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-10:
            continue  # degenerate (collinear points)
        normal = normal / norm_len

        # Ensure normal points upward (positive Z component)
        if normal[2] < 0:
            normal = -normal

        # Reject if tilted beyond the allowed limit
        if normal[2] < cos_max_tilt:
            continue

        d = -np.dot(normal, p1)
        distances = np.abs(subsample @ normal + d)
        inlier_count = np.sum(distances < threshold)

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_normal = normal
            best_d = d

        if (i + 1) % 200 == 0:
            print(f"    RANSAC iteration {i+1}/{n_iter}, "
                  f"best inliers: {best_inlier_count:,}")

    if best_normal is None:
        raise RuntimeError(
            "RANSAC failed to find any valid plane within tilt constraint")

    # Evaluate the winner on the full cloud
    full_distances = np.abs(points @ best_normal + best_d)
    inlier_mask = full_distances < threshold

    print(f"  RANSAC result: {np.sum(inlier_mask):,} inliers "
          f"({100 * np.mean(inlier_mask):.1f}%)")
    print(f"  Initial normal: [{best_normal[0]:.6f}, "
          f"{best_normal[1]:.6f}, {best_normal[2]:.6f}]")
    return best_normal, best_d, inlier_mask


def refine_plane(points, inlier_mask):
    """
    Refine a plane fit using SVD on the inlier points.

    Computes the centroid of the inliers, centres them, and takes the
    right-singular vector corresponding to the smallest singular value
    as the refined plane normal.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    inlier_mask : ndarray, shape (N,), dtype bool

    Returns
    -------
    normal : ndarray, shape (3,)
        Refined unit normal (upward-pointing).
    d : float
        Offset term of the refined plane equation.
    """
    inlier_points = points[inlier_mask]
    centroid = inlier_points.mean(axis=0)
    centered = inlier_points - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    normal = Vt[2]  # direction of least variance

    if normal[2] < 0:
        normal = -normal

    d = -np.dot(normal, centroid)
    print(f"  Refined normal: [{normal[0]:.6f}, {normal[1]:.6f}, "
          f"{normal[2]:.6f}]")
    print(f"  Refined d: {d:.6f}")
    return normal, d


def build_rotation_matrix(normal):
    """
    Build a 3x3 rotation matrix that maps *normal* to [0, 0, 1].

    Uses Rodrigues' rotation formula via ``scipy.spatial.transform``.

    Parameters
    ----------
    normal : ndarray, shape (3,)
        The ground-plane normal (unit vector, pointing upward).

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.  Apply as ``rotated = points @ R.T``.
    """
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    axis_len = np.linalg.norm(axis)

    if axis_len < 1e-10:
        return np.eye(3)  # already aligned

    axis = axis / axis_len
    angle = np.arccos(np.clip(np.dot(normal, target), -1.0, 1.0))
    R = Rotation.from_rotvec(angle * axis).as_matrix()

    print(f"  Rotation angle: {np.degrees(angle):.3f} deg")
    print(f"  Rotation axis:  [{axis[0]:.6f}, {axis[1]:.6f}, "
          f"{axis[2]:.6f}]")
    return R


def rotate_and_translate(points, R, inlier_mask):
    """
    Rotate the pointcloud and shift so the ground sits at Z = 0.

    Parameters
    ----------
    points : ndarray, shape (N, 3)
    R : ndarray, shape (3, 3)
        Rotation matrix from ``build_rotation_matrix``.
    inlier_mask : ndarray, shape (N,), dtype bool
        Ground-plane inlier mask.

    Returns
    -------
    rotated : ndarray, shape (N, 3), dtype float32
        Rotated and Z-shifted pointcloud.
    """
    rotated = (points @ R.T).astype(np.float32)
    z_offset = np.mean(rotated[inlier_mask, 2])
    rotated[:, 2] -= z_offset
    print(f"  Z offset applied: {z_offset:.4f}")
    return rotated


def compute_rect_density(x, y, x_edges, y_edges):
    """
    Compute inverse-square-compensated point density on a rectangular grid.

    For each bin, the raw point count is multiplied by r^2 (squared
    distance from the scanner origin to the bin centre) to counteract
    the natural 1/r^2 density falloff of a spinning LiDAR.  The result
    is divided by the bin area to yield a density with units
    ``points * m`` (count * m^2 / m^2, after the r^2 factor).

    Parameters
    ----------
    x, y : ndarray
        Point coordinates within this slice.
    x_edges, y_edges : ndarray
        Histogram bin edges.

    Returns
    -------
    density : ndarray, shape (n_x_bins, n_y_bins)
        Compensated density.
    counts : ndarray, same shape
        Raw (uncompensated) bin counts.
    """
    counts, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    x_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centres = 0.5 * (y_edges[:-1] + y_edges[1:])
    Xc, Yc = np.meshgrid(x_centres, y_centres, indexing='ij')
    r_sq = np.maximum(Xc**2 + Yc**2, 1.0)  # clamp near origin

    dx = x_edges[1] - x_edges[0]
    dy = y_edges[1] - y_edges[0]
    density = counts * r_sq / (dx * dy)

    return density, counts


def compute_scan_bounds(rotated_points, keep_pct=95.0):
    """
    Compute a tight bounding box via percentile trimming.

    For each axis independently, the lower and upper tails are clipped
    so that *keep_pct* % of points remain.  This removes sparse
    outliers and reveals the dense scan region.

    Parameters
    ----------
    rotated_points : ndarray, shape (N, 3)
    keep_pct : float
        Percentage of points to keep (default 95 %).

    Returns
    -------
    bounds : dict
        ``{'X': (lo, hi), 'Y': (lo, hi), 'Z': (lo, hi)}``.
    """
    tail = (100.0 - keep_pct) / 2.0
    bounds = {}
    for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
        lo = float(np.percentile(rotated_points[:, axis], tail))
        hi = float(np.percentile(rotated_points[:, axis], 100.0 - tail))
        bounds[name] = (lo, hi)
        print(f"  {name}: [{lo:+.2f}, {hi:+.2f}] (range {hi - lo:.2f} m)")
    return bounds


def generate_html_viewer(rotated_points, intensity, output_path,
                         bounds=None, max_points=300_000):
    """
    Generate an interactive 3D HTML viewer of the rotated pointcloud.

    Uses Plotly Scatter3d.  If *bounds* is provided, points outside
    the bounding box are cropped before display.  The cloud is randomly
    subsampled to *max_points* for browser performance.

    Parameters
    ----------
    rotated_points : ndarray, shape (N, 3)
    intensity : ndarray, shape (N,)
    output_path : str
        Path for the output ``.html`` file.
    bounds : dict or None
        Output of ``compute_scan_bounds()``.
    max_points : int
        Maximum points to display (default 300 000).
    """
    n_orig = len(rotated_points)

    if bounds is not None:
        mask = np.ones(n_orig, dtype=bool)
        for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
            if name in bounds:
                lo, hi = bounds[name]
                mask &= ((rotated_points[:, axis] >= lo) &
                         (rotated_points[:, axis] <= hi))
        rotated_points = rotated_points[mask]
        intensity = intensity[mask]
        print(f"  Cropped to {len(rotated_points):,} points "
              f"({100 * len(rotated_points) / n_orig:.1f}% of {n_orig:,})")

    rng = np.random.default_rng(123)
    n = len(rotated_points)
    idx = rng.choice(n, size=min(n, max_points), replace=False)
    pts = rotated_points[idx]

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=1, color=pts[:, 2], colorscale='Turbo',
            cmin=np.percentile(pts[:, 2], 1),
            cmax=np.percentile(pts[:, 2], 99),
            colorbar=dict(title='Z (m)'), opacity=0.6,
        ),
        hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
    )])
    fig.update_layout(
        title='Rotated Pointcloud (ground = z=0, auto-cropped)',
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)',
                   zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    fig.write_html(output_path)
    print(f"  HTML viewer saved to {output_path}")
    print(f"  ({len(idx):,} points displayed)")


# ============================================================
# INTERACTIVE CROP SELECTION (local HTTP server)
# ============================================================

class SessionState:
    """
    Thread-safe shared state between the HTTP handler and the background
    heatmap-generation worker.

    Attributes
    ----------
    lock : threading.Lock
        Protects all mutable fields during concurrent access.
    status : str
        One of ``"idle"``, ``"running"``, ``"done"``, ``"error"``.
    progress_current, progress_total : int
        Slice counters for the progress bar.
    error_message : str
        Non-empty when ``status == "error"``.
    run_count : int
        Incremented each time the user clicks Go.  Allows the background
        thread to detect that a newer run has superseded it.
    done_event : threading.Event
        Signalled by ``GET /quit`` to tell ``main()`` the session is over.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"
        self.progress_current = 0
        self.progress_total = 0
        self.error_message = ""
        self.run_count = 0
        self.done_event = threading.Event()


class CropParamHandler(http.server.BaseHTTPRequestHandler):
    """
    HTTP request handler for the interactive crop-selection UI.

    Class-level attributes are set before the server starts:

    - ``html_content`` : str — the full HTML page to serve on ``GET /``
    - ``session`` : SessionState — shared state for progress + images
    - ``rotated_points`` : ndarray — full rotated pointcloud
    - ``cfg_template`` : dict — config dict (RANSAC metadata included)
    """
    html_content = ""
    session = None
    rotated_points = None
    cfg_template = None

    # ------ helpers ------

    def _respond(self, code, content_type, body):
        """Send an HTTP response with the given code, type, and body bytes."""
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond_json(self, obj):
        """Serialize *obj* as JSON and send a 200 application/json response."""
        body = json.dumps(obj).encode("utf-8")
        self._respond(200, "application/json", body)

    # ------ GET routes ------

    def do_GET(self):
        if self.path == "/":
            body = self.html_content.encode("utf-8")
            self._respond(200, "text/html; charset=utf-8", body)

        elif self.path == "/status":
            s = self.session
            with s.lock:
                payload = {
                    "status": s.status,
                    "progress": [s.progress_current, s.progress_total],
                    "run_id": s.run_count,
                    "error": s.error_message,
                }
            self._respond_json(payload)

        elif self.path == "/quit":
            self._respond_json({"status": "ok"})
            self.session.done_event.set()

        else:
            self.send_error(404)

    # ------ POST routes ------

    def do_POST(self):
        if self.path == "/go":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            params = json.loads(body)

            s = self.session

            # Reject if a run is already in progress
            with s.lock:
                if s.status == "running":
                    self._respond_json({
                        "status": "error",
                        "message": "Generation already in progress",
                    })
                    return

            # Validate: ensure min < max, clamp bin_size
            for lo_key, hi_key in [("x_min", "x_max"), ("y_min", "y_max"),
                                   ("z_min", "z_max")]:
                if params[lo_key] > params[hi_key]:
                    params[lo_key], params[hi_key] = (params[hi_key],
                                                       params[lo_key])
            params["bin_size"] = max(
                params.get("bin_size", DEFAULT_BIN_SIZE), 0.001)
            params["slice_thickness"] = max(
                params.get("slice_thickness", DEFAULT_SLICE_THICKNESS), 0.001)

            # Reset session state for new run
            with s.lock:
                s.status = "running"
                s.progress_current = 0
                s.progress_total = 0
                s.error_message = ""
                s.run_count += 1
                run_id = s.run_count

            # Launch background generation
            t = threading.Thread(
                target=CropParamHandler._run_generation,
                args=(params, run_id),
                daemon=True,
            )
            t.start()

            self._respond_json({"status": "started", "run_id": run_id})
        else:
            self.send_error(404)

    # ------ background worker ------

    @staticmethod
    def _run_generation(params, run_id):
        """Entry point for the background heatmap-generation thread."""
        s = CropParamHandler.session
        try:
            generate_heatmaps_incremental(
                CropParamHandler.rotated_points,
                CropParamHandler.cfg_template,
                params, s, run_id,
            )
        except Exception as e:
            with s.lock:
                s.status = "error"
                s.error_message = str(e)
            import traceback
            traceback.print_exc()

    def log_message(self, format, *args):
        pass  # suppress per-request console noise


def build_interactive_html(plotly_fragment, scan_bounds):
    """
    Build the full HTML page for the interactive crop-selection viewer.

    The page shows a 3D Plotly scatter plot with crop-parameter controls
    and a Go button.  A progress overlay appears while heatmaps are
    being generated in Python.  When generation finishes, Python opens
    the standalone ``heatmap_slices.html`` viewer in a new tab, and
    the 3D view re-enables so the user can adjust and generate again.

    A **Done** button (or Ctrl+C in the terminal) ends the session.

    Parameters
    ----------
    plotly_fragment : str
        HTML+JS fragment from ``fig.to_html(full_html=False)``.
    scan_bounds : dict
        Auto-crop bounds ``{{'X': (lo, hi), 'Y': (lo, hi), 'Z': (lo, hi)}}``.

    Returns
    -------
    html : str
        Complete HTML document string.
    """
    x_lo, x_hi = scan_bounds['X']
    y_lo, y_hi = scan_bounds['Y']

    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Ground Plane - Interactive Analysis</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #111; color: #eee; font-family: sans-serif;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
.plot-area {{ flex: 1; min-height: 0; }}
.plot-area > div {{ width: 100%; height: 100%; }}
.controls {{ background: #222; padding: 14px 24px; flex-shrink: 0;
             box-shadow: 0 -2px 8px rgba(0,0,0,0.5);
             display: flex; flex-wrap: wrap; align-items: center;
             gap: 10px 20px; }}
.controls label {{ font-size: 13px; color: #bbb; display: flex;
                   align-items: center; gap: 4px; }}
.controls input[type=number] {{ width: 80px; padding: 4px 6px;
       background: #333; color: #eee; border: 1px solid #555;
       border-radius: 3px; font-size: 13px; }}
.controls .sep {{ width: 1px; height: 28px; background: #444; }}
#goBtn {{ font-size: 16px; padding: 8px 28px; cursor: pointer;
          background: #e67e22; color: #fff; border: none;
          border-radius: 5px; font-weight: bold; margin-left: auto; }}
#goBtn:hover {{ background: #d35400; }}
#goBtn:disabled {{ background: #555; color: #999; cursor: default; }}
#doneBtn {{ font-size: 14px; padding: 6px 18px; cursor: pointer;
            background: #c0392b; color: #fff; border: none;
            border-radius: 5px; }}
#doneBtn:hover {{ background: #a93226; }}
.hint {{ font-size: 11px; color: #666; text-align: center;
         padding: 4px; flex-shrink: 0; }}

/* ===== Progress Overlay ===== */
#progressOverlay {{ display: none; position: fixed; top: 0; left: 0;
                    width: 100%; height: 100%;
                    background: rgba(0,0,0,0.75);
                    justify-content: center; align-items: center;
                    z-index: 9999; }}
.progress-box {{ background: #222; border-radius: 10px; padding: 30px 50px;
                 text-align: center; min-width: 350px;
                 box-shadow: 0 4px 20px rgba(0,0,0,0.6); }}
.progress-text {{ font-size: 18px; margin-bottom: 16px; }}
.progress-bar-track {{ height: 20px; background: #333; border-radius: 10px;
                       overflow: hidden; }}
.progress-bar-fill {{ height: 100%; background: #e67e22; width: 0%;
                      transition: width 0.3s ease; }}
#progressDetail {{ margin-top: 10px; font-size: 14px; color: #aaa; }}
</style></head><body>

<div class="plot-area" id="plotContainer">
{plotly_fragment}
</div>

<div class="controls">
  <label>X min <input type="number" id="x_min" value="{x_lo:.2f}" step="0.1"></label>
  <label>X max <input type="number" id="x_max" value="{x_hi:.2f}" step="0.1"></label>
  <div class="sep"></div>
  <label>Y min <input type="number" id="y_min" value="{y_lo:.2f}" step="0.1"></label>
  <label>Y max <input type="number" id="y_max" value="{y_hi:.2f}" step="0.1"></label>
  <div class="sep"></div>
  <label>Z min <input type="number" id="z_min" value="{DEFAULT_SLICE_MIN}" step="0.1"></label>
  <label>Z max <input type="number" id="z_max" value="{DEFAULT_SLICE_MAX}" step="0.1"></label>
  <div class="sep"></div>
  <label>Bin size (m) <input type="number" id="bin_size" value="{DEFAULT_BIN_SIZE}" step="0.001" min="0.001"></label>
  <label>Slice thickness (m) <input type="number" id="slice_thickness" value="{DEFAULT_SLICE_THICKNESS}" step="0.005" min="0.001"></label>
  <button id="goBtn">Go &mdash; Generate Heatmaps</button>
  <button id="doneBtn">Done</button>
</div>
<div class="hint">Rotate / zoom the 3D view above, then set crop bounds and click Go.
     Heatmap viewer will open in a new tab. Adjust and click Go again for a different region.</div>

<!-- ========== PROGRESS OVERLAY ========== -->
<div id="progressOverlay">
  <div class="progress-box">
    <div class="progress-text">Generating heatmaps&hellip;</div>
    <div class="progress-bar-track">
      <div class="progress-bar-fill" id="progressFill"></div>
    </div>
    <div id="progressDetail">Starting&hellip;</div>
  </div>
</div>

<script>
/* ---------- Plotly resize ---------- */
(function() {{
  var container = document.getElementById('plotContainer');
  var plotDiv = container.querySelector('.plotly-graph-div');
  if (plotDiv) {{
    plotDiv.style.width = '100%';
    plotDiv.style.height = '100%';
    window.dispatchEvent(new Event('resize'));
  }}
}})();

/* ---------- State ---------- */
var pollTimer = null;
var currentRunId = 0;

/* ---------- Go button ---------- */
document.getElementById('goBtn').addEventListener('click', function() {{
  var btn = this;
  var params = {{
    x_min: parseFloat(document.getElementById('x_min').value),
    x_max: parseFloat(document.getElementById('x_max').value),
    y_min: parseFloat(document.getElementById('y_min').value),
    y_max: parseFloat(document.getElementById('y_max').value),
    z_min: parseFloat(document.getElementById('z_min').value),
    z_max: parseFloat(document.getElementById('z_max').value),
    bin_size: parseFloat(document.getElementById('bin_size').value),
    slice_thickness: parseFloat(document.getElementById('slice_thickness').value),
  }};
  btn.disabled = true;
  btn.textContent = 'Starting\\u2026';

  fetch('/go', {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(params),
  }})
  .then(function(r) {{ return r.json(); }})
  .then(function(data) {{
    if (data.status === 'error') {{
      alert('Error: ' + data.message);
      btn.disabled = false;
      btn.textContent = 'Go \\u2014 Generate Heatmaps';
      return;
    }}
    currentRunId = data.run_id;
    document.getElementById('progressOverlay').style.display = 'flex';
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressDetail').textContent = 'Starting\\u2026';
    pollTimer = setInterval(pollStatus, 1500);
  }})
  .catch(function(err) {{
    alert('Error communicating with Python: ' + err);
    btn.disabled = false;
    btn.textContent = 'Go \\u2014 Generate Heatmaps';
  }});
}});

/* ---------- Polling ---------- */
function pollStatus() {{
  fetch('/status')
  .then(function(r) {{ return r.json(); }})
  .then(function(data) {{
    if (data.run_id !== currentRunId) return;

    var pct = data.progress[1] > 0
        ? Math.round(100 * data.progress[0] / data.progress[1]) : 0;
    document.getElementById('progressFill').style.width = pct + '%';
    document.getElementById('progressDetail').textContent =
        data.progress[0] + ' / ' + data.progress[1] + ' slices';

    if (data.status === 'done') {{
      clearInterval(pollTimer);
      pollTimer = null;
      document.getElementById('progressOverlay').style.display = 'none';
      document.getElementById('goBtn').disabled = false;
      document.getElementById('goBtn').textContent = 'Go \\u2014 Generate Heatmaps';
    }} else if (data.status === 'error') {{
      clearInterval(pollTimer);
      pollTimer = null;
      document.getElementById('progressOverlay').style.display = 'none';
      alert('Generation error: ' + data.error);
      document.getElementById('goBtn').disabled = false;
      document.getElementById('goBtn').textContent = 'Go \\u2014 Generate Heatmaps';
    }}
  }});
}}

/* ---------- Done button ---------- */
document.getElementById('doneBtn').addEventListener('click', function() {{
  if (confirm('End the analysis session?')) {{
    fetch('/quit').then(function() {{
      document.body.innerHTML =
        '<div style="display:flex;justify-content:center;' +
        'align-items:center;height:100vh;background:#111;color:#eee;' +
        'font-family:sans-serif"><h2>Session ended. You can close ' +
        'this tab.</h2></div>';
    }});
  }}
}});
</script>
</body></html>"""


def run_interactive_session(rotated_points, intensity, scan_bounds, cfg,
                            max_points=300_000):
    """
    Start the persistent interactive analysis session.

    Opens a browser with a 3D pointcloud viewer and crop controls.
    The user can click **Go** to generate heatmaps (which appear
    automatically in the browser), then click **Back** to adjust
    parameters and generate again — as many times as they like.

    The session ends when the user clicks **Done** in the browser
    or presses Ctrl+C in the terminal.

    Parameters
    ----------
    rotated_points : ndarray, shape (N, 3)
    intensity : ndarray, shape (N,)
    scan_bounds : dict
        Auto-crop bounds from ``compute_scan_bounds()``.
    cfg : dict
        Configuration dictionary.  Must include ``_``-prefixed RANSAC
        metadata for report writing.
    max_points : int
        Maximum points to display in the 3D viewer (subsampled).
    """
    # Crop and subsample for the Plotly figure
    n_orig = len(rotated_points)
    mask = np.ones(n_orig, dtype=bool)
    for axis, name in [(0, 'X'), (1, 'Y'), (2, 'Z')]:
        if name in scan_bounds:
            lo, hi = scan_bounds[name]
            mask &= ((rotated_points[:, axis] >= lo) &
                     (rotated_points[:, axis] <= hi))
    cropped = rotated_points[mask]

    rng = np.random.default_rng(123)
    n = len(cropped)
    idx = rng.choice(n, size=min(n, max_points), replace=False)
    pts = cropped[idx]

    fig = go.Figure(data=[go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        mode='markers',
        marker=dict(
            size=1, color=pts[:, 2], colorscale='Turbo',
            cmin=np.percentile(pts[:, 2], 1),
            cmax=np.percentile(pts[:, 2], 99),
            colorbar=dict(title='Z (m)'), opacity=0.6,
        ),
        hovertemplate='x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<extra></extra>',
    )])
    fig.update_layout(
        scene=dict(xaxis_title='X (m)', yaxis_title='Y (m)',
                   zaxis_title='Z (m)', aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=10),
    )

    plotly_fragment = fig.to_html(full_html=False, include_plotlyjs='cdn')
    html = build_interactive_html(plotly_fragment, scan_bounds)

    # Configure the handler
    session = SessionState()
    CropParamHandler.html_content = html
    CropParamHandler.session = session
    CropParamHandler.rotated_points = rotated_points
    CropParamHandler.cfg_template = cfg

    # Start server on a free port
    server = http.server.HTTPServer(('127.0.0.1', 0), CropParamHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/"
    print(f"  Interactive viewer at {url}")
    print(f"  (If it doesn't open automatically, paste this URL "
          f"into your browser)")
    print(f"  Click Go to generate heatmaps.  Click Done or "
          f"press Ctrl+C to end.")
    webbrowser.open(url)

    try:
        session.done_event.wait()
    except KeyboardInterrupt:
        print("\n  Interrupted.")

    server.shutdown()
    print("  Session ended.")


def generate_heatmaps(rotated_points, cfg):
    """
    Generate rectangular heatmap slices and a click-through HTML viewer.

    For each Z-slice, a 2D histogram is computed on the X-Y plane,
    compensated by r^2, and rendered as a matplotlib image.  All images
    are collected into a single-page HTML viewer with instant image
    swapping for easy visual comparison across heights.

    Parameters
    ----------
    rotated_points : ndarray, shape (N, 3)
    cfg : dict
        Configuration dictionary from ``prompt_for_config()``.

    Returns
    -------
    slice_stats : list of (z_low, z_high, n_pts, skipped) tuples
    """
    import base64
    from io import BytesIO

    heatmap_dir = cfg["HEATMAP_DIR"]
    output_dir = cfg["OUTPUT_DIR"]
    os.makedirs(heatmap_dir, exist_ok=True)

    # Determine bounding box  (None → use data extent)
    x_lo = cfg["SLICE_X_MIN"] if cfg["SLICE_X_MIN"] is not None else float(np.min(rotated_points[:, 0]))
    x_hi = cfg["SLICE_X_MAX"] if cfg["SLICE_X_MAX"] is not None else float(np.max(rotated_points[:, 0]))
    y_lo = cfg["SLICE_Y_MIN"] if cfg["SLICE_Y_MIN"] is not None else float(np.min(rotated_points[:, 1]))
    y_hi = cfg["SLICE_Y_MAX"] if cfg["SLICE_Y_MAX"] is not None else float(np.max(rotated_points[:, 1]))

    x_range = x_hi - x_lo
    y_range = y_hi - y_lo
    bin_size = cfg["BIN_SIZE"]

    n_x_bins = max(1, int(round(x_range / bin_size)))
    n_y_bins = max(1, int(round(y_range / bin_size)))

    x_edges = np.linspace(x_lo, x_hi, n_x_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_y_bins + 1)
    print(f"  Grid: {n_x_bins} x {n_y_bins} bins, "
          f"bin size {bin_size * 100:.1f} cm")
    print(f"  X: [{x_lo:.2f}, {x_hi:.2f}], Y: [{y_lo:.2f}, {y_hi:.2f}]")

    # Figure size: landscape, Y horizontal, X vertical
    fig_w = 14
    fig_h = max(3, fig_w * (x_range / y_range)) + 1.0

    # Filter points to the X-Y bounding box
    mask = ((rotated_points[:, 0] >= x_lo) &
            (rotated_points[:, 0] <= x_hi) &
            (rotated_points[:, 1] >= y_lo) &
            (rotated_points[:, 1] <= y_hi))
    pts = rotated_points[mask]
    print(f"  Points in box: {len(pts):,} of {len(rotated_points):,}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    slices = np.arange(cfg["SLICE_MIN"], cfg["SLICE_MAX"],
                       cfg["SLICE_THICKNESS"])
    slice_stats = []
    html_images = []  # (label, n_pts, base64_png)

    for i, z_low in enumerate(slices):
        z_high = z_low + cfg["SLICE_THICKNESS"]
        in_slice = (z >= z_low) & (z < z_high)
        n_pts = int(np.sum(in_slice))

        label = f"z=[{z_low:+.2f}, {z_high:+.2f})"
        print(f"  Slice {i+1}/{len(slices)}: {label}, "
              f"{n_pts:,} points", end="")

        if n_pts < 10:
            print(" -- skipped")
            slice_stats.append((z_low, z_high, n_pts, True))
            continue
        print()

        density, counts = compute_rect_density(
            x[in_slice], y[in_slice], x_edges, y_edges)

        # Flip X so it increases upward in the image
        img = density[::-1, :]

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        # Lock subplot position — prevents jostling between frames
        fig.subplots_adjust(left=0.10, right=0.88, bottom=0.12, top=0.90)
        vmax = (np.percentile(density[density > 0], 99)
                if np.any(density > 0) else 1)
        im = ax.imshow(img, extent=[y_lo, y_hi, x_lo, x_hi],
                       cmap='hot', vmin=0, vmax=vmax,
                       aspect='equal', interpolation='nearest')
        ax.set_xlabel('Y (m)', fontsize=11)
        ax.set_ylabel('X (m)', fontsize=11)
        ax.set_title(f"Compensated Density   {label}   "
                     f"({n_pts:,} pts)", fontsize=12)
        plt.colorbar(im, ax=ax, label='Density (pts*m)',
                     shrink=0.7, pad=0.02)

        # Save high-res PNG
        fname = f"slice_{i:02d}_z_{z_low:+.2f}_to_{z_high:+.2f}.png"
        fig.savefig(os.path.join(heatmap_dir, fname), dpi=cfg["DPI"])

        # Capture lower-res base64 for the HTML viewer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        buf.close()
        plt.close(fig)

        html_images.append((label, n_pts, b64))
        slice_stats.append((z_low, z_high, n_pts, False))

    # ---- Build single-image click-through HTML viewer ----
    html_path = os.path.join(output_dir, "heatmap_slices.html")
    labels_json = json.dumps([lbl for lbl, _, _ in html_images])
    pts_json = json.dumps([n for _, n, _ in html_images])

    with open(html_path, 'w') as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8">\n')
        f.write('<title>Ground Plane Heatmap Slices</title>\n')
        f.write('<style>\n')
        f.write("""
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #eee; font-family: sans-serif;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
.toolbar { background: #222; padding: 10px 20px; display: flex;
           align-items: center; gap: 12px; flex-shrink: 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.5); }
.toolbar button { font-size: 18px; padding: 6px 16px; cursor: pointer;
                  background: #444; color: #eee; border: 1px solid #666;
                  border-radius: 4px; min-width: 44px; }
.toolbar button:hover { background: #555; }
.toolbar button:active { background: #666; }
#label { font-size: 15px; font-weight: bold; min-width: 260px; text-align: center; }
#pts { font-size: 13px; color: #aaa; min-width: 100px; }
.slider-row { background: #1a1a1a; padding: 6px 20px; display: flex;
              align-items: center; gap: 12px; flex-shrink: 0; }
#scrubber { flex: 1; height: 20px; cursor: pointer; accent-color: #f80; }
#pos { font-size: 13px; color: #aaa; min-width: 70px; text-align: right; }
.viewer { flex: 1; display: flex; justify-content: center; align-items: center;
          overflow: hidden; padding: 5px; }
.viewer img { max-width: 100%; max-height: 100%; object-fit: contain; }
.help { font-size: 11px; color: #666; text-align: center; padding: 4px;
        flex-shrink: 0; }
""")
        f.write('</style></head><body>\n')

        # Toolbar
        f.write('<div class="toolbar">\n')
        f.write('  <button id="prevBtn" onclick="go(cur-1)">'
                '&larr; Prev</button>\n')
        f.write('  <span id="label"></span>\n')
        f.write('  <span id="pts"></span>\n')
        f.write('  <button id="nextBtn" onclick="go(cur+1)">'
                'Next &rarr;</button>\n')
        f.write('</div>\n')

        # Scrubber
        f.write('<div class="slider-row">\n')
        f.write(f'  <input type="range" id="scrubber" min="0" '
                f'max="{len(html_images) - 1}" value="0" '
                f'oninput="go(parseInt(this.value))">\n')
        f.write('  <span id="pos"></span>\n')
        f.write('</div>\n')

        # Image display
        f.write('<div class="viewer"><img id="display" /></div>\n')
        f.write('<div class="help">Arrow keys / Space to step | '
                'Home / End to jump | Drag slider to scrub</div>\n')

        # JavaScript: preloaded data-URI array + navigation logic
        f.write('<script>\n')
        f.write(f'const labels = {labels_json};\n')
        f.write(f'const pts = {pts_json};\n')
        f.write(f'const N = {len(html_images)};\n')
        f.write('const imgs = [\n')
        for idx, (_, _, b64) in enumerate(html_images):
            comma = ',' if idx < len(html_images) - 1 else ''
            f.write(f'  "data:image/png;base64,{b64}"{comma}\n')
        f.write('];\n')
        f.write("""
let cur = 0;
function go(i) {
  cur = Math.max(0, Math.min(i, N-1));
  document.getElementById('display').src = imgs[cur];
  document.getElementById('label').textContent = labels[cur];
  document.getElementById('pts').textContent = pts[cur].toLocaleString() + ' pts';
  document.getElementById('scrubber').value = cur;
  document.getElementById('pos').textContent = (cur+1) + ' / ' + N;
}
go(0);
document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ')
    { e.preventDefault(); go(cur+1); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp')
    { e.preventDefault(); go(cur-1); }
  else if (e.key === 'Home') { e.preventDefault(); go(0); }
  else if (e.key === 'End') { e.preventDefault(); go(N-1); }
});
""")
        f.write('</script></body></html>\n')

    print(f"  Viewer HTML saved to {html_path}")
    print(f"  ({len(html_images)} slices)")
    return slice_stats


def generate_heatmaps_incremental(rotated_points, cfg_template, crop_params,
                                   session, run_id):
    """
    Generate heatmap slices with incremental progress reporting.

    This is the background-thread version of :func:`generate_heatmaps`.
    It merges *crop_params* into a local copy of *cfg_template*, then
    generates slices one-by-one with progress reporting.

    PNGs and the standalone ``heatmap_slices.html`` are saved to disk,
    ``write_report()`` is called, and then the heatmap viewer HTML is
    automatically opened in a new browser tab.

    Parameters
    ----------
    rotated_points : ndarray, shape (N, 3)
    cfg_template : dict
        Config dict with RANSAC metadata stored under ``_``-prefixed keys.
    crop_params : dict
        Keys: x_min, x_max, y_min, y_max, z_min, z_max, bin_size,
        slice_thickness.
    session : SessionState
    run_id : int
        The ``session.run_count`` at the time this run was started.
        If a newer run supersedes us, we abort.
    """
    import base64
    from io import BytesIO
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend, safe for threads

    # Merge crop parameters into a working copy of the config
    cfg = dict(cfg_template)
    cfg["SLICE_X_MIN"] = crop_params["x_min"]
    cfg["SLICE_X_MAX"] = crop_params["x_max"]
    cfg["SLICE_Y_MIN"] = crop_params["y_min"]
    cfg["SLICE_Y_MAX"] = crop_params["y_max"]
    cfg["SLICE_MIN"] = crop_params["z_min"]
    cfg["SLICE_MAX"] = crop_params["z_max"]
    cfg["BIN_SIZE"] = crop_params["bin_size"]
    cfg["SLICE_THICKNESS"] = crop_params["slice_thickness"]

    heatmap_dir = cfg["HEATMAP_DIR"]
    output_dir = cfg["OUTPUT_DIR"]
    os.makedirs(heatmap_dir, exist_ok=True)

    # Bounding box
    x_lo = cfg["SLICE_X_MIN"]
    x_hi = cfg["SLICE_X_MAX"]
    y_lo = cfg["SLICE_Y_MIN"]
    y_hi = cfg["SLICE_Y_MAX"]

    x_range = x_hi - x_lo
    y_range = y_hi - y_lo
    bin_size = cfg["BIN_SIZE"]

    n_x_bins = max(1, int(round(x_range / bin_size)))
    n_y_bins = max(1, int(round(y_range / bin_size)))

    x_edges = np.linspace(x_lo, x_hi, n_x_bins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_y_bins + 1)
    print(f"  [Run {run_id}] Grid: {n_x_bins} x {n_y_bins} bins, "
          f"bin size {bin_size * 100:.1f} cm")
    print(f"  [Run {run_id}] X: [{x_lo:.2f}, {x_hi:.2f}], "
          f"Y: [{y_lo:.2f}, {y_hi:.2f}]")

    # Figure size: landscape, Y horizontal, X vertical
    fig_w = 14
    fig_h = max(3, fig_w * (x_range / y_range)) + 1.0

    # Filter points to the X-Y bounding box
    mask = ((rotated_points[:, 0] >= x_lo) &
            (rotated_points[:, 0] <= x_hi) &
            (rotated_points[:, 1] >= y_lo) &
            (rotated_points[:, 1] <= y_hi))
    pts = rotated_points[mask]
    print(f"  [Run {run_id}] Points in box: {len(pts):,} "
          f"of {len(rotated_points):,}")

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]

    slices = np.arange(cfg["SLICE_MIN"], cfg["SLICE_MAX"],
                       cfg["SLICE_THICKNESS"])

    with session.lock:
        session.progress_total = len(slices)
        session.progress_current = 0

    slice_stats = []
    html_images = []

    for i, z_low in enumerate(slices):
        # Check if a newer run has superseded us
        with session.lock:
            if session.run_count != run_id:
                print(f"  [Run {run_id}] Superseded by run {session.run_count}"
                      f" — aborting.")
                return

        z_high = z_low + cfg["SLICE_THICKNESS"]
        in_slice = (z >= z_low) & (z < z_high)
        n_pts = int(np.sum(in_slice))

        label = f"z=[{z_low:+.2f}, {z_high:+.2f})"
        print(f"  [Run {run_id}] Slice {i+1}/{len(slices)}: {label}, "
              f"{n_pts:,} points", end="")

        if n_pts < 10:
            print(" -- skipped")
            slice_stats.append((z_low, z_high, n_pts, True))
            with session.lock:
                session.progress_current = i + 1
            continue
        print()

        density, counts = compute_rect_density(
            x[in_slice], y[in_slice], x_edges, y_edges)

        # Flip X so it increases upward in the image
        img = density[::-1, :]

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.subplots_adjust(left=0.10, right=0.88, bottom=0.12, top=0.90)
        vmax = (np.percentile(density[density > 0], 99)
                if np.any(density > 0) else 1)
        im = ax.imshow(img, extent=[y_lo, y_hi, x_lo, x_hi],
                       cmap='hot', vmin=0, vmax=vmax,
                       aspect='equal', interpolation='nearest')
        ax.set_xlabel('Y (m)', fontsize=11)
        ax.set_ylabel('X (m)', fontsize=11)
        ax.set_title(f"Compensated Density   {label}   "
                     f"({n_pts:,} pts)", fontsize=12)
        plt.colorbar(im, ax=ax, label='Density (pts*m)',
                     shrink=0.7, pad=0.02)

        # Save high-res PNG to disk
        fname = f"slice_{i:02d}_z_{z_low:+.2f}_to_{z_high:+.2f}.png"
        fig.savefig(os.path.join(heatmap_dir, fname), dpi=cfg["DPI"])

        # Capture lower-res base64 for the HTML viewer
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=120)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('ascii')
        buf.close()
        plt.close(fig)

        with session.lock:
            session.progress_current = i + 1

        html_images.append((label, n_pts, b64))
        slice_stats.append((z_low, z_high, n_pts, False))

    # ---- Write standalone heatmap_slices.html to disk ----
    html_path = os.path.join(output_dir, "heatmap_slices.html")
    labels_json = json.dumps([lbl for lbl, _, _ in html_images])
    pts_json = json.dumps([n for _, n, _ in html_images])

    with open(html_path, 'w') as f:
        f.write('<!DOCTYPE html><html><head><meta charset="utf-8">\n')
        f.write('<title>Ground Plane Heatmap Slices</title>\n')
        f.write('<style>\n')
        f.write("""
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #111; color: #eee; font-family: sans-serif;
       display: flex; flex-direction: column; height: 100vh; overflow: hidden; }
.toolbar { background: #222; padding: 10px 20px; display: flex;
           align-items: center; gap: 12px; flex-shrink: 0;
           box-shadow: 0 2px 8px rgba(0,0,0,0.5); }
.toolbar button { font-size: 18px; padding: 6px 16px; cursor: pointer;
                  background: #444; color: #eee; border: 1px solid #666;
                  border-radius: 4px; min-width: 44px; }
.toolbar button:hover { background: #555; }
.toolbar button:active { background: #666; }
#label { font-size: 15px; font-weight: bold; min-width: 260px; text-align: center; }
#pts { font-size: 13px; color: #aaa; min-width: 100px; }
.slider-row { background: #1a1a1a; padding: 6px 20px; display: flex;
              align-items: center; gap: 12px; flex-shrink: 0; }
#scrubber { flex: 1; height: 20px; cursor: pointer; accent-color: #f80; }
#pos { font-size: 13px; color: #aaa; min-width: 70px; text-align: right; }
.viewer { flex: 1; display: flex; justify-content: center; align-items: center;
          overflow: hidden; padding: 5px; }
.viewer img { max-width: 100%; max-height: 100%; object-fit: contain; }
.help { font-size: 11px; color: #666; text-align: center; padding: 4px;
        flex-shrink: 0; }
""")
        f.write('</style></head><body>\n')
        f.write('<div class="toolbar">\n')
        f.write('  <button id="prevBtn" onclick="go(cur-1)">'
                '&larr; Prev</button>\n')
        f.write('  <span id="label"></span>\n')
        f.write('  <span id="pts"></span>\n')
        f.write('  <button id="nextBtn" onclick="go(cur+1)">'
                'Next &rarr;</button>\n')
        f.write('</div>\n')
        f.write('<div class="slider-row">\n')
        f.write(f'  <input type="range" id="scrubber" min="0" '
                f'max="{len(html_images) - 1}" value="0" '
                f'oninput="go(parseInt(this.value))">\n')
        f.write('  <span id="pos"></span>\n')
        f.write('</div>\n')
        f.write('<div class="viewer"><img id="display" /></div>\n')
        f.write('<div class="help">Arrow keys / Space to step | '
                'Home / End to jump | Drag slider to scrub</div>\n')
        f.write('<script>\n')
        f.write(f'const labels = {labels_json};\n')
        f.write(f'const pts = {pts_json};\n')
        f.write(f'const N = {len(html_images)};\n')
        f.write('const imgs = [\n')
        for idx_img, (_, _, b64_val) in enumerate(html_images):
            comma = ',' if idx_img < len(html_images) - 1 else ''
            f.write(f'  "data:image/png;base64,{b64_val}"{comma}\n')
        f.write('];\n')
        f.write("""
let cur = 0;
function go(i) {
  cur = Math.max(0, Math.min(i, N-1));
  document.getElementById('display').src = imgs[cur];
  document.getElementById('label').textContent = labels[cur];
  document.getElementById('pts').textContent = pts[cur].toLocaleString() + ' pts';
  document.getElementById('scrubber').value = cur;
  document.getElementById('pos').textContent = (cur+1) + ' / ' + N;
}
go(0);
document.addEventListener('keydown', function(e) {
  if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ')
    { e.preventDefault(); go(cur+1); }
  else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp')
    { e.preventDefault(); go(cur-1); }
  else if (e.key === 'Home') { e.preventDefault(); go(0); }
  else if (e.key === 'End') { e.preventDefault(); go(N-1); }
});
""")
        f.write('</script></body></html>\n')

    print(f"  [Run {run_id}] Viewer HTML saved to {html_path}")
    print(f"  [Run {run_id}] {len(html_images)} slices generated.")

    # Write report
    write_report(cfg, cfg["_normal"], cfg["_d"],
                 cfg["_n_inliers"], cfg["_n_total"],
                 cfg["_rotation_angle_deg"], cfg["_rotation_axis"],
                 cfg["_z_offset"], slice_stats)
    print(f"  [Run {run_id}] Report saved.")

    # Open heatmap viewer in a new browser tab
    if html_images:
        webbrowser.open('file://' + os.path.abspath(html_path))

    with session.lock:
        session.status = "done"


def write_report(cfg, normal, d, n_inliers, n_total,
                 rotation_angle_deg, rotation_axis, z_offset,
                 slice_stats):
    """
    Write a human-readable summary report.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.
    normal : ndarray, shape (3,)
    d : float
    n_inliers, n_total : int
    rotation_angle_deg : float
    rotation_axis : ndarray, shape (3,)
    z_offset : float
    slice_stats : list of tuples
    """
    report_path = os.path.join(cfg["OUTPUT_DIR"], "ground_plane_report.txt")
    with open(report_path, 'w') as f:
        f.write("Ground Plane Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input: {cfg['INPUT_CSV']}\n")
        f.write(f"Points loaded: {n_total:,}\n\n")

        f.write("Ground Plane Detection (RANSAC)\n")
        f.write(f"  Normal vector: [{normal[0]:.6f}, {normal[1]:.6f}, "
                f"{normal[2]:.6f}]\n")
        f.write(f"  Plane equation: {normal[0]:.6f}*x + "
                f"{normal[1]:.6f}*y + {normal[2]:.6f}*z + "
                f"{d:.6f} = 0\n")
        f.write(f"  d: {d:.6f}\n")
        f.write(f"  Inlier count: {n_inliers:,} "
                f"({100 * n_inliers / n_total:.1f}%)\n")
        f.write(f"  Inlier threshold: {cfg['RANSAC_THRESHOLD']} m\n")
        f.write(f"  RANSAC iterations: {cfg['RANSAC_ITERATIONS']}\n")
        f.write(f"  Max tilt constraint: {cfg['MAX_TILT_DEG']} deg\n\n")

        f.write("Rotation\n")
        f.write(f"  Rotation angle: {rotation_angle_deg:.3f} deg\n")
        f.write(f"  Rotation axis: [{rotation_axis[0]:.6f}, "
                f"{rotation_axis[1]:.6f}, {rotation_axis[2]:.6f}]\n")
        f.write(f"  Z offset (ground shift to z=0): {z_offset:.4f}\n\n")

        f.write("Heatmap Configuration\n")
        f.write(f"  Slice thickness: {cfg['SLICE_THICKNESS']} m\n")
        f.write(f"  Z range: {cfg['SLICE_MIN']} to {cfg['SLICE_MAX']} m\n")
        f.write(f"  X bounds: [{cfg['SLICE_X_MIN']}, "
                f"{cfg['SLICE_X_MAX']}]\n")
        f.write(f"  Y bounds: [{cfg['SLICE_Y_MIN']}, "
                f"{cfg['SLICE_Y_MAX']}]\n")
        f.write(f"  Bin size: {cfg['BIN_SIZE'] * 100:.1f} cm\n\n")

        f.write("Slice Summary\n")
        n_generated = sum(1 for s in slice_stats if not s[3])
        n_skipped = sum(1 for s in slice_stats if s[3])
        f.write(f"  Total slices: {len(slice_stats)}\n")
        f.write(f"  Generated: {n_generated}\n")
        f.write(f"  Skipped (< 10 points): {n_skipped}\n\n")

        for z_low, z_high, n_pts, skipped in slice_stats:
            status = "SKIPPED" if skipped else "OK"
            f.write(f"  z=[{z_low:+.2f}, {z_high:+.2f}): "
                    f"{n_pts:>10,} points  [{status}]\n")

    print(f"  Report saved to {report_path}")


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Entry point.  Prompts the user for RANSAC parameters, then runs the
    full pipeline: load -> RANSAC -> rotate -> auto-crop -> persistent
    interactive browser session (3D viewer, crop controls, Go button,
    heatmap viewer, Back button for multiple rounds).
    """
    cfg = prompt_for_config()

    t0 = time.time()
    os.makedirs(cfg["HEATMAP_DIR"], exist_ok=True)

    # ---- Step 1: Load ----
    print("Step 1: Loading pointcloud...")
    xyz, intensity, timestamp = load_pointcloud(cfg["INPUT_CSV"])
    n_total = len(xyz)

    # ---- Step 2: RANSAC ground-plane detection (full cloud) ----
    print("\nStep 2: RANSAC ground plane detection...")
    normal, d, inlier_mask = ransac_plane_fit(
        xyz, cfg["RANSAC_ITERATIONS"], cfg["RANSAC_THRESHOLD"],
        cfg["RANSAC_SUBSAMPLE"], cfg["MAX_TILT_DEG"])

    print("  Refining plane with SVD on inliers...")
    normal, d = refine_plane(xyz, inlier_mask)

    inlier_mask = np.abs(xyz @ normal + d) < cfg["RANSAC_THRESHOLD"]
    n_inliers = int(np.sum(inlier_mask))
    print(f"  Final inlier count: {n_inliers:,} "
          f"({100 * n_inliers / n_total:.1f}%)")

    # ---- Step 3: Rotation ----
    print("\nStep 3: Rotating pointcloud to align ground with z=0...")
    R = build_rotation_matrix(normal)
    rotation_axis = np.cross(normal, [0, 0, 1])
    axis_len = np.linalg.norm(rotation_axis)
    if axis_len > 1e-10:
        rotation_axis = rotation_axis / axis_len
        rotation_angle_deg = np.degrees(
            np.arccos(np.clip(normal[2], -1, 1)))
    else:
        rotation_axis = np.array([0.0, 0.0, 1.0])
        rotation_angle_deg = 0.0

    rotated = rotate_and_translate(xyz, R, inlier_mask)
    z_offset_val = float(np.mean((xyz @ R.T)[inlier_mask, 2]))

    del xyz  # free memory

    # ---- Step 3.5: Auto-crop bounds ----
    print("\nStep 3.5: Computing auto-crop bounds...")
    print(f"  Auto-crop: keeping {cfg['AUTO_CROP_PERCENTILE']}% "
          f"of points per axis")
    scan_bounds = compute_scan_bounds(
        rotated, keep_pct=cfg["AUTO_CROP_PERCENTILE"])

    # Save static 3D viewer HTML for archival
    viewer_path = os.path.join(cfg["OUTPUT_DIR"],
                               "rotated_pointcloud_viewer.html")
    generate_html_viewer(rotated, intensity, viewer_path,
                         bounds=scan_bounds)

    # Store RANSAC metadata in cfg so the background thread can
    # write reports without needing these variables directly.
    cfg["_normal"] = normal
    cfg["_d"] = d
    cfg["_n_inliers"] = n_inliers
    cfg["_n_total"] = n_total
    cfg["_rotation_angle_deg"] = rotation_angle_deg
    cfg["_rotation_axis"] = rotation_axis
    cfg["_z_offset"] = z_offset_val

    # ---- Step 4: Interactive session ----
    print("\nStep 4: Opening interactive session in browser...")
    print("  Adjust crop parameters and click Go to generate heatmaps.")
    print("  You can go back and regenerate with different parameters.")
    print("  Click 'Done' or press Ctrl+C when finished.\n")
    run_interactive_session(rotated, intensity, scan_bounds, cfg)

    elapsed = time.time() - t0
    print(f"\nDone! Total time: {elapsed:.1f}s")
    print(f"Output directory: {cfg['OUTPUT_DIR']}")


if __name__ == "__main__":
    main()
