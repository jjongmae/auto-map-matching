"""
회전 행렬 및 오일러 각 변환을 위한 기하학 유틸리티
"""
import numpy as np
import math


def isRotationMatrix(R):
    """행렬이 유효한 회전 행렬인지 확인합니다."""
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def euler_to_R_cam(yaw, pitch, roll):
    """
    vaid_gis.geometry.cam_rotation과 일치합니다:
    Roll(z) -> Pitch(x) -> Yaw(y)
    여기서 yaw_cv = -yaw, pitch_cv = -pitch 입니다.
    """
    # 라디안으로 변환
    y_cv = math.radians(-yaw)
    p_cv = math.radians(-pitch)
    r_cv = math.radians(roll)

    # R_roll (Z축 회전)
    R_z = np.array([
        [math.cos(r_cv), -math.sin(r_cv), 0],
        [math.sin(r_cv),  math.cos(r_cv), 0],
        [0,              0,               1]
    ])

    # R_pitch (X축 회전)
    R_x = np.array([
        [1, 0,              0],
        [0, math.cos(p_cv), -math.sin(p_cv)],
        [0, math.sin(p_cv),  math.cos(p_cv)]
    ])

    # R_yaw (Y축 회전)
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
    euler_to_R_cam의 역함수입니다.
    R = Rz(r) * Rx(p') * Ry(y')로 분해합니다.
    yaw, pitch, roll을 도(degree) 단위로 반환합니다.
    """
    # R[2, 1] = sin(p')
    # p' = asin(R[2, 1])
    # pitch_cv = p'
    # pitch = -pitch_cv

    sy = math.sqrt(R[0, 1] * R[0, 1] + R[1, 1] * R[1, 1])
    singular = sy < 1e-6

    if not singular:
        # p' = atan2(R[2,1], sy) # asin보다 더 강건함
        p_cv = math.atan2(R[2, 1], sy)

        # r (roll)은 R[0,1], R[1,1]에서 유래
        # R[0,1] = -s_z * c_x
        # R[1,1] =  c_z * c_x
        # tan(z) = -R[0,1] / R[1,1]
        r_cv = math.atan2(-R[0, 1], R[1, 1])

        # y' (yaw)는 R[2,0], R[2,2]에서 유래
        # R[2,0] = -c_x * s_y
        # R[2,2] =  c_x * c_y
        # tan(y) = -R[2,0] / R[2,2]
        y_cv = math.atan2(-R[2, 0], R[2, 2])
    else:
        # 짐벌 락: pitch가 +/- 90도 (cos(p') = 0)
        # R[2,1] ~ +/- 1
        p_cv = math.atan2(R[2, 1], sy)

        # r_cv - y_cv (또는 +) 모호성. y_cv = 0으로 가정
        y_cv = 0
        r_cv = math.atan2(R[1, 0], R[0, 0])

    # 도(degree) 변환 및 부호 반전
    yaw = -math.degrees(y_cv)
    pitch = -math.degrees(p_cv)
    roll = math.degrees(r_cv)

    # 반환 전 정규화
    yaw = normalize_angle(yaw)
    roll = normalize_angle(roll)
    # pitch는 R_cam_to_euler 로직상 이미 -90~90 범위에 있음 (atan2 사용)

    return yaw, pitch, roll


def normalize_angle(angle):
    """각도를 -180 ~ 180 범위로 정규화합니다."""
    while angle > 180:
        angle -= 360
    while angle <= -180:
        angle += 360
    return angle


def eulerAnglesToRotationMatrix(yaw, pitch, roll):
    """
    오일러 각(도)을 회전 행렬로 변환합니다.
    순서: Yaw(Z) -> Pitch(Y) -> Roll(X)
    """
    # 도를 라디안으로 변환
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
    회전 행렬을 오일러 각(도)으로 변환합니다.
    반환: (yaw, pitch, roll)
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
