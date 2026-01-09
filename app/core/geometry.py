"""
Geometry utilities for rotation matrix and Euler angle conversions
"""
import numpy as np
import math


def isRotationMatrix(R):
    """Checks if a matrix is a valid rotation matrix."""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def euler_to_R_cam(yaw, pitch, roll):
    """
    Matches vaid_gis.geometry.cam_rotation:
    Roll(z) -> Pitch(x) -> Yaw(y)
    with yaw_cv = -yaw, pitch_cv = -pitch.
    """
    # Convert to radians
    y_cv = math.radians(-yaw)
    p_cv = math.radians(-pitch)
    r_cv = math.radians(roll)

    # R_roll (Z)
    R_z = np.array([
        [math.cos(r_cv), -math.sin(r_cv), 0],
        [math.sin(r_cv),  math.cos(r_cv), 0],
        [0,              0,               1]
    ])

    # R_pitch (X)
    R_x = np.array([
        [1, 0,              0],
        [0, math.cos(p_cv), -math.sin(p_cv)],
        [0, math.sin(p_cv),  math.cos(p_cv)]
    ])

    # R_yaw (Y)
    R_y = np.array([
        [ math.cos(y_cv), 0, math.sin(y_cv)],
        [ 0,              1, 0],
        [-math.sin(y_cv), 0, math.cos(y_cv)]
    ])

    # R = Rz @ Rx @ Ry
    R = np.dot(R_z, np.dot(R_x, R_y))
    return R


def R_cam_to_euler(R):
    """
    Inverse of euler_to_R_cam.
    Decomposes R = Rz(r) * Rx(p') * Ry(y')
    Returns yaw, pitch, roll in degrees.
    """
    # R[2, 1] = sin(p')
    # p' = asin(R[2, 1])
    # pitch_cv = p'
    # pitch = -pitch_cv

    sy = math.sqrt(R[0, 1] * R[0, 1] + R[1, 1] * R[1, 1])
    singular = sy < 1e-6

    if not singular:
        # p' = atan2(R[2,1], sy) # more robust than asin
        p_cv = math.atan2(R[2, 1], sy)

        # r (roll) comes from R[0,1], R[1,1]
        # R[0,1] = -s_z * c_x
        # R[1,1] =  c_z * c_x
        # tan(z) = -R[0,1] / R[1,1]
        r_cv = math.atan2(-R[0, 1], R[1, 1])

        # y' (yaw) comes from R[2,0], R[2,2]
        # R[2,0] = -c_x * s_y
        # R[2,2] =  c_x * c_y
        # tan(y) = -R[2,0] / R[2,2]
        y_cv = math.atan2(-R[2, 0], R[2, 2])
    else:
        # Gimbal lock: pitch is +/- 90 (cos(p') = 0)
        # R[2,1] ~ +/- 1
        p_cv = math.atan2(R[2, 1], sy)

        # r_cv - y_cv (or +) ambiguity. Assume y_cv = 0
        y_cv = 0
        r_cv = math.atan2(R[1, 0], R[0, 0])

    # deg conversion and sign flip
    yaw = -math.degrees(y_cv)
    pitch = -math.degrees(p_cv)
    roll = math.degrees(r_cv)

    return yaw, pitch, roll


def eulerAnglesToRotationMatrix(yaw, pitch, roll):
    """
    Convert Euler angles (degrees) to rotation matrix.
    Order: Yaw(Z) -> Pitch(Y) -> Roll(X)
    """
    # Convert degrees to radians
    yaw_rad = math.radians(yaw)
    pitch_rad = math.radians(pitch)
    roll_rad = math.radians(roll)

    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll_rad), -math.sin(roll_rad)],
                    [0, math.sin(roll_rad), math.cos(roll_rad)]])

    R_y = np.array([[math.cos(pitch_rad), 0, math.sin(pitch_rad)],
                    [0, 1, 0],
                    [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]])

    R_z = np.array([[math.cos(yaw_rad), -math.sin(yaw_rad), 0],
                    [math.sin(yaw_rad), math.cos(yaw_rad), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def rotationMatrixToEulerAngles(R):
    """
    Convert rotation matrix to Euler angles (degrees).
    Returns: (yaw, pitch, roll)
    """
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2]) # roll
        y = math.atan2(-R[2,0], sy)    # pitch
        z = math.atan2(R[1,0], R[0,0]) # yaw
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.degrees(z), np.degrees(y), np.degrees(x) # Yaw, Pitch, Roll
