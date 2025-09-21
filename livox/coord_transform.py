import numpy as np
import struct
import pandas as pd

def euler_to_matrix(pitch, roll, yaw):
    """Convert roll, pitch, yaw (radians) to a 3x3 rotation matrix."""
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_p = np.array([
        [1, 0, 0],
        [0, cp, -sp],
        [0, sp, cp]
    ])

    R_r = np.array([
        [cr, 0, sr],
        [0, 1, 0],
        [-sr, 0, cr]
    ])

    R_y = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return R_y @ R_r @ R_p

def transform_point(point_sensor, roll, pitch, yaw, t_vec):
    """Apply extrinsic transform to a point (in meters)."""
    R = euler_to_matrix(roll, pitch, yaw)
    return R @ point_sensor + t_vec

data_frame = pd.read_csv("test/7test.csv")

#Example float values from Device Info Block
rot_list = [np.deg2rad(-0.05000000074505806), np.deg2rad(8.770000457763672), np.deg2rad(-174.13999938964844)]
t_vec = np.array([12.345000267028809, -23.45599937438965, 34.56700134277344])

for row in data_frame.iterrows():
    row = row[1]
    pt_vec = np.array([row["x1"], row["y1"],row["z1"]])/1000
    print(transform_point(pt_vec, rot_list[0], rot_list[1], rot_list[2], t_vec))
    break


# Pitch:  (-0.05000000074505806,)
# roll:  (8.770000457763672,)
# yaw:  (-174.13999938964844,)
# x:  1095075103
# y:  -1044666909
# z:  1107969180
#(12.345000267028809,)
#y:  (-23.45599937438965,)
#z:  (34.56700134277344,)
# 10.9774, -23.5331, 33.3097, 3
# print(int_to_float(1095075103) - 1544/1000)
# pt_sensor = np.array([1544,-62,-1034])/1000
# rot_list = [np.deg2rad(8.770000457763672), np.deg2rad(-0.05000000074505806), np.deg2rad(-174.13999938964844)]
# #pitch roll yaw
# print(transform_point(pt_sensor, np.deg2rad(-0.05000000074505806),np.deg2rad(8.770000457763672),  np.deg2rad(-174.13999938964844), 1095075103, -1044666909, 1107969180))