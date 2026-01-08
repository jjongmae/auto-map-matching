"""
카메라 파라미터 모델
"""
from dataclasses import dataclass

@dataclass
class CameraParams:
    """
    카메라의 모든 파라미터를 저장하는 데이터 클래스.
    사전(dict) 대신 이 클래스를 사용하면 타입 안정성과 자동 완성을 통해
    개발 편의성을 높일 수 있습니다.
    """
    # Extrinsic Parameters (월드 좌표계 기준)
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0

    # Intrinsic Parameters (픽셀 단위)
    fx: float = 1000.0
    fy: float = 1000.0
    cx: float = 960.0
    cy: float = 540.0

    # Distortion Coefficients
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    # Clipping Planes
    near: float = 0.1
    far: float = 10000.0  # 10km까지 투영 가능

    # Resolution
    resolution_width: int = 1920
    resolution_height: int = 1080
