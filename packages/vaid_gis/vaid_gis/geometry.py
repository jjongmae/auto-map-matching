"""
지오메트리 및 좌표 변환 관련 유틸리티 함수.
"""
import numpy as np
import math

# 축 재배열: ENU(X 동, Y 북, Z 상) → 카메라 CV(X 오른쪽, Y 아래, Z 전방)
B_ENU2CV = np.array([
    [ 1,  0,  0],   # X_world →  X_cam
    [ 0,  0, -1],   # Y_world → -Y_cam  (북은 화면 위)
    [ 0,  1,  0]    # Z_world →  Y_cam  (상은 화면 아래)
], dtype=np.float32)

def cam_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Roll-Z → Pitch-X → Yaw-Y (내재적)"""
    # 부호 변환 ───────────────────────────
    yaw_cv   = -yaw    # 우→좌
    pitch_cv = -pitch  # 들기→숙이기
    roll_cv  =  roll   # 동일

    ry, rx, rz = map(math.radians, (yaw_cv, pitch_cv, roll_cv))

    R_yaw = np.array([
        [ math.cos(ry), 0, math.sin(ry)],
        [ 0,            1, 0           ],
        [-math.sin(ry), 0, math.cos(ry)]
    ])
    R_pitch = np.array([
        [1, 0,             0           ],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx),  math.cos(rx)]
    ])
    R_roll = np.array([
        [ math.cos(rz), -math.sin(rz), 0],
        [ math.sin(rz),  math.cos(rz), 0],
        [ 0,            0,             1]
    ])
    return R_roll @ R_pitch @ R_yaw

def distort(x, y, k1=0, k2=0, p1=0, p2=0, k3=0):
    """카메라 왜곡 모델을 적용합니다."""
    r2 = x*x + y*y
    radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
    x_d = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    y_d = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    return x_d, y_d

def undistort(x_d, y_d, k1=0, k2=0, p1=0, p2=0, k3=0, iterations=5):
    """
    왜곡된 좌표를 반복적으로 보정하여 원본 정규 좌표를 추정합니다.
    """
    x_n, y_n = x_d, y_d
    for _ in range(iterations):
        r2 = x_n*x_n + y_n*y_n
        k_inv = 1 / (1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2)
        x_n = (x_d - (2*p1*x_n*y_n + p2*(r2 + 2*x_n*x_n))) * k_inv
        y_n = (y_d - (p1*(r2 + 2*y_n*y_n) + 2*p2*x_n*y_n)) * k_inv
    return x_n, y_n

def ray_mesh_intersect(ray_origin, ray_direction, triangles):
    """Möller-Trumbore 알고리즘으로 광선-메시 교차점을 찾습니다. (벡터화 및 최적화)"""
    EPSILON = 1e-6
    
    # 모든 삼각형에 대한 계산 준비
    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    pvec = np.cross(ray_direction, edge2)
    det = np.sum(edge1 * pvec, axis=1)

    # 1. 평행 필터링
    parallel_mask = np.abs(det) < EPSILON
    
    # 유효한 삼각형들만 남김
    valid_mask = ~parallel_mask
    if not np.any(valid_mask): return None

    det = det[valid_mask]
    edge1 = edge1[valid_mask]
    edge2 = edge2[valid_mask]
    pvec = pvec[valid_mask]
    valid_triangles = triangles[valid_mask]

    tvec = ray_origin - valid_triangles[:, 0]

    # 2. U 좌표 필터링
    u = np.sum(tvec * pvec, axis=1) / det
    u_mask = (u >= 0) & (u <= 1)
    if not np.any(u_mask): return None

    det = det[u_mask]
    tvec = tvec[u_mask]
    edge1 = edge1[u_mask]
    edge2 = edge2[u_mask]
    u = u[u_mask]

    # 3. V 좌표 필터링
    qvec = np.cross(tvec, edge1)
    v = np.sum(ray_direction * qvec, axis=1) / det
    v_mask = (v >= 0) & (u + v <= 1)
    if not np.any(v_mask): return None

    det = det[v_mask]
    edge2 = edge2[v_mask]
    qvec = qvec[v_mask]

    # 4. T 좌표 계산 및 필터링
    t = np.sum(edge2 * qvec, axis=1) / det
    t_mask = t > EPSILON
    
    intersecting_t = t[t_mask]
    if len(intersecting_t) == 0: return None

    # 가장 가까운 교차점 반환
    return ray_origin + intersecting_t.min() * ray_direction

def point_to_segment_dist(p, v, w):
    """점 p와 선분 vw 사이의 최단 거리를 계산합니다."""
    l2 = np.sum((v - w)**2)
    if l2 == 0.0: return np.linalg.norm(p - v)
    t = max(0, min(1, np.dot(p - v, w - v) / l2))
    projection = v + t * (w - v)
    return np.linalg.norm(p - projection)

def point_to_segments_dist_vectorized(p, v_array, w_array):
    """점 p와 여러 선분들 사이의 최단 거리를 벡터화하여 계산합니다.

    기존 point_to_segment_dist 함수와 정확히 동일한 로직을 벡터화하여 구현.

    Args:
        p: 점 좌표 (2,)
        v_array: 선분 시작점들 (N, 2)
        w_array: 선분 끝점들 (N, 2)

    Returns:
        distances: 각 선분까지의 거리 배열 (N,)
    """
    # 기존: l2 = np.sum((v - w)**2)
    l2 = np.sum((v_array - w_array)**2, axis=1)  # (N,)

    # 기존: if l2 == 0.0: return np.linalg.norm(p - v)
    zero_length_mask = l2 == 0.0

    # 결과 배열 초기화
    distances = np.zeros(len(v_array))

    # 길이가 0인 선분들 처리
    if np.any(zero_length_mask):
        distances[zero_length_mask] = np.linalg.norm(p - v_array[zero_length_mask], axis=1)

    # 일반 선분들 처리
    if np.any(~zero_length_mask):
        valid_v = v_array[~zero_length_mask]
        valid_w = w_array[~zero_length_mask]
        valid_l2 = l2[~zero_length_mask]

        # 기존: t = max(0, min(1, np.dot(p - v, w - v) / l2))
        pv = p - valid_v  # (M, 2)
        wv = valid_w - valid_v  # (M, 2)
        dot_products = np.sum(pv * wv, axis=1)  # (M,)
        t = np.clip(dot_products / valid_l2, 0, 1)  # (M,)

        # 기존: projection = v + t * (w - v)
        projections = valid_v + t[:, np.newaxis] * wv  # (M, 2)

        # 기존: return np.linalg.norm(p - projection)
        distances[~zero_length_mask] = np.linalg.norm(p - projections, axis=1)

    return distances
