# import numpy as np
# import cv2
# from scipy.spatial.transform import Rotation as R

# data = np.load('../camera_params/camera_pose_inv.npz')
# Rot_cam_to_chess = data['R_inv']
# t_cam_in_chess = data['tvec_inv']

# # Convert Euler angles to rotation matrix
# euler_angles = [0.94, -41.58, -0.8]  # [roll, pitch, yaw] in degrees
# # r_additional = R.from_euler('zyx', euler_angles, degrees=True)
# # print("zyx", r_additional.as_matrix())
# r_additional = R.from_euler('ZYX', euler_angles, degrees=True)
# # print("ZYX", r_additional.as_matrix())
# # r_additional = R.from_euler('xyz', euler_angles, degrees=True)
# # print("xyz", r_additional.as_matrix())
# # r_additional = R.from_euler('XYZ', euler_angles, degrees=True)
# # print("XYZ", r_additional.as_matrix())

# R_ee_base = r_additional.as_matrix()

# T_ee_base = np.array([[117.76], [0.83], [779.84]]) # in mm unit

# # Compute the new rotation and translation matrices
# R_total = R_ee_base @ Rot_cam_to_chess
# t_total = T_ee_base + R_ee_base @ np.array([[0], [0], [20]]) + R_ee_base @ t_cam_in_chess

# print('R_total', R_total)
# print('t_total', t_total)
# # Save the new rotation and translation matrices
# np.savez('../camera_params/camera_pose_base.npz', R=R_total, tvec=t_total)

# print("New rotation and translation matrices have been saved to '../camera_params/camera_pose_base.npz'")


import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

def make_homogeneous_transform(R_mat, t_vec):
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = t_vec.flatten()
    return T

# -------------------------------------------------------------------
# 1) End-Effector -> Base
euler_angles = [67.01, -26.58, 1.08]  # [roll, pitch, yaw] in degrees
r_additional = R.from_euler('ZYX', euler_angles, degrees=True)
R_ee_base = r_additional.as_matrix()

# translation (mm)
T_ee_base = np.array([[-234.9], [-547.19], [983.86]])  # 3x1

# Build 4x4: EE->Base
T_ee_base_4x4 = make_homogeneous_transform(R_ee_base, T_ee_base)

# Compute Base->EE by inverting
# T_base_ee_4x4 = inv(T_ee_base_4x4)

# -------------------------------------------------------------------
# 2) Camera -> Chessboard
data = np.load('../camera_params/camera_pose_inv.npz')
Rot_cam_to_chess = data['R_inv']       # 3x3
t_cam_in_chess   = data['tvec_inv']    # 3x1

# Build 4x4: Camera->Chessboard
T_cam_chess_4x4 = make_homogeneous_transform(Rot_cam_to_chess, t_cam_in_chess)

# We want Chess->Camera, so invert
# T_chess_cam_4x4 = inv(T_cam_chess_4x4)

# -------------------------------------------------------------------
# 3) Chessboard offset in EE frame
chess_thickness_offset = np.array([[0], [0], [20]])  # 20 mm along EE's Z
T_chess_ee_4x4 = make_homogeneous_transform(np.eye(3), chess_thickness_offset)

# -------------------------------------------------------------------
# 4) Final composition: Base->Cam
T_cam_base_4x4 = T_ee_base_4x4 @ T_chess_ee_4x4 @ T_cam_chess_4x4

# Extract rotation and translation
R_cam_base = T_cam_base_4x4[:3, :3]
t_cam_base = T_cam_base_4x4[:3, 3]

print("R_cam_base:\n", R_cam_base)
print("t_cam_base:\n", t_cam_base)

# Optionally save
np.savez('../camera_params/camera_pose_base.npz',
         R=R_cam_base,
         tvec=t_cam_base)
print("Saved camera pose in base frame to '../camera_params/camera_pose_base.npz'")
