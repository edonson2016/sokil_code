import open3d as o3d
import numpy as np
from get_curv import get_curv
from cluster_curv import curv_pt
from extract_reflec import read_pcd_with_reflectivity
from rotate import rotate
from density import heat_map

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Surface Detection')
    parser.add_argument('--filename', type=str, help='Rel File path to point cloud')
    parser.add_argument('--HEATMAP', type=bool, default=True, help='Run Heatmap analysis or curvature analysis')
    parser.add_argument('--VIS', type=bool, default=False, help='Run visualizations in addition to analysis')
    return parser.parse_args()

# Starting parameters...
args = parse_args()
filename = args.filename
HEATMAP = args.HEATMAP
VIS = args.VIS

pcd_raw = o3d.io.read_point_cloud(filename)
pcd_raw = rotate(pcd_raw)

if not HEATMAP:
    plane_model, inliers = pcd_raw.segment_plane(
            distance_threshold=0.1,  
            ransac_n=10,
            num_iterations=1000
        )
    pcd_raw = pcd_raw.select_by_index(inliers, invert = True)

    print("Plane seg")

intensity_raw = np.array(read_pcd_with_reflectivity(filename)["intensity"]).reshape(-1, 1).astype(np.float64)

pcd_raw_pt = np.asarray(pcd_raw.points)
pcd = o3d.t.geometry.PointCloud()
pcd.point["positions"] = o3d.core.Tensor(pcd_raw_pt, dtype=o3d.core.Dtype.Float64)
pcd.point["intensity"] = o3d.core.Tensor(intensity_raw, dtype=o3d.core.Dtype.Float64)
pcd = pcd.voxel_down_sample(0.025)
intensity = pcd.point["intensity"].numpy().reshape(-1)

max_in = max(intensity)
min_in = min(intensity)
normalized_in = (intensity - min_in) / (max_in - min_in)
colors_raw = np.zeros((len(normalized_in), 3), dtype=np.float64)
colors_raw[:, 0] = normalized_in
colors_raw[:, 2] = 1 - normalized_in
# Assign colors
pcd.point["colors"] = o3d.core.Tensor(colors_raw, dtype=o3d.core.Dtype.Float64)

if VIS:
    o3d.visualization.draw([pcd])

pcd_pt = pcd.point["positions"].numpy()

max_z = np.max(pcd_pt[:,2])
min_z = np.min(pcd_pt[:,2])
max_x = np.max(pcd_pt[:,0])
min_x = np.min(pcd_pt[:,0])

x_box_ct = 20
z_box_ct = 20
z_len = max_z-min_z
x_len = max_x-min_x

curr_pos = (0, 0, 0)

print("Start Loop")

for i in range(x_box_ct):
    for j in range(z_box_ct):
        
        pos = pcd.point["positions"]

        x = pos[:, 0]
        z = pos[:, 2]

        mask = (z > curr_pos[2]) & \
            (z < curr_pos[2] + z_len / z_box_ct) & \
            (x > curr_pos[0]) & \
            (x < curr_pos[0] + x_len / x_box_ct)

        cut_pcd_i = mask.nonzero()[0].numpy().tolist()
        if len(cut_pcd_i) > 10000:
            curr_pcd = pcd.select_by_index(cut_pcd_i)

            if len(list(curr_pcd.point["positions"])) >= 10000:
                curr_pcd.paint_uniform_color([0.9, 0.2, 0.2])
                pcd.paint_uniform_color([0.0, 0.0, 0.0])
                o3d.visualization.draw_geometries([pcd.to_legacy(), curr_pcd.to_legacy()])

                if HEATMAP:
                    heat_map(curr_pcd, y_cutoff=1.5, radius=0.15)
                else:
                    intensity = curr_pcd.point["intensity"].numpy().reshape(-1)

                    max_in = max(intensity)
                    min_in = min(intensity)
                    normalized_in = (intensity - min_in) / (max_in - min_in)
                    colors_raw = np.zeros((len(normalized_in), 3), dtype=np.float64)
                    colors_raw[:, 0] = normalized_in
                    colors_raw[:, 2] = 1 - normalized_in
                    # Assign colors
                    pcd.point["colors"] = o3d.core.Tensor(colors_raw, dtype=o3d.core.Dtype.Float64)
                    curr_pcd_leg = curr_pcd.to_legacy()
                    curr_pcd_leg.colors = o3d.utility.Vector3dVector(colors_raw)
                    # Assign colors
                    o3d.visualization.draw_geometries([curr_pcd_leg])

                    curv_pcd = get_curv(curr_pcd)
                    curv_pt(pcd, curv_pcd)
                
        curr_pos = (curr_pos[0], 0, curr_pos[2] + z_len/z_box_ct)
    curr_pos = (curr_pos[0]+x_len/x_box_ct, 0, min_z)

