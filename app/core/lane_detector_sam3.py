"""
SAM3를 사용한 차선 검출 모듈

Ultralytics SAM3 모델을 사용하여 이미지에서 차선을 세그먼테이션합니다.
참고: https://docs.ultralytics.com/models/sam-3/
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

from app.core.skeleton_lane_processor import SkeletonLaneProcessor, SkeletonProcessorConfig


class LaneDetectorSAM3:
    """SAM3 기반 차선 검출기"""

    def __init__(
        self,
        model_path: str = "models/sam3.pt",
        device: str = "cuda",
        conf: float = 0.5,
        use_skeleton: bool = False,
        skeleton_config: Optional[SkeletonProcessorConfig] = None
    ):
        """
        Args:
            model_path: SAM3 모델 파일 경로
            device: cuda 또는 cpu
            conf: 신뢰도 임계값 (기본 0.5)
            use_skeleton: 스켈레톤 기반 후처리 사용 여부 (기본 False)
            skeleton_config: 스켈레톤 프로세서 설정 (use_skeleton=True일 때만 적용)
        """
        self.model_path = model_path
        self.device = device
        self.conf = conf
        self.use_skeleton = use_skeleton
        self.predictor = None
        self.model_loaded = False

        # 스켈레톤 프로세서 초기화
        if self.use_skeleton:
            self.skeleton_processor = SkeletonLaneProcessor(skeleton_config)
        else:
            self.skeleton_processor = None

    def load_model(self) -> bool:
        """SAM3 모델 로드"""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=self.conf,
                task="segment",
                mode="predict",
                model=self.model_path,
                imgsz=1540,  # 입력 해상도 설정 (letterbox 자동 처리)
                half=True,
                device=self.device,
                save=False  # runs/segment 폴더 생성 방지
            )
            self.predictor = SAM3SemanticPredictor(overrides=overrides)
            self.model_loaded = True
            print(f"[SAM3] 모델 로드 완료: {self.model_path}")
            return True

        except ImportError as e:
            print(f"[SAM3] ultralytics 패키지 오류: {e}")
            print("[SAM3] pip install -U ultralytics (8.3.237 이상 필요)")
            return False
        except Exception as e:
            print(f"[SAM3] 모델 로드 실패: {e}")
            return False

    def detect_lanes(
        self,
        image_path: str,
        text_prompts: List[str] = None,
        pixel_interval: int = 20,
        poly_degree: int = 2
    ) -> List[Dict]:
        """
        이미지에서 차선을 검출합니다.

        Args:
            image_path: 이미지 파일 경로
            text_prompts: 텍스트 프롬프트 (기본: ["lane", "road marking"])
            pixel_interval: 점 샘플링 간격 (픽셀 단위, 기본 20px)
            poly_degree: 다항식 피팅 차수 (0이면 피팅 안함, 기본 2차, 곡선 도로는 3차 권장)

        Returns:
            차선 리스트 [{"id": int, "points": [[x,y],...], "mask": np.ndarray}, ...]
        """
        if not self.model_loaded:
            if not self.load_model():
                return []

        if text_prompts is None:
            text_prompts = ["lane", "road marking", "lane line"]

        try:
            self.predictor.set_image(image_path)
            results = self.predictor(text=text_prompts)

            if not results or len(results) == 0:
                print("[SAM3] 차선을 찾지 못했습니다")
                return []

            lanes = self._process_results(results, pixel_interval, poly_degree)
            if self.use_skeleton:
                print(f"[SAM3] {len(lanes)}개의 차선 검출됨 (스켈레톤 후처리)")
            else:
                print(f"[SAM3] {len(lanes)}개의 차선 검출됨 (다항식 피팅: {poly_degree}차)")
            return lanes

        except Exception as e:
            print(f"[SAM3] 검출 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _process_results(self, results, pixel_interval: int, poly_degree: int = 2) -> List[Dict]:
        """SAM3 결과를 차선 데이터로 변환"""
        lanes = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            # 스켈레톤 기반 후처리 사용 시
            if self.use_skeleton and self.skeleton_processor is not None:
                skeleton_lanes = self.skeleton_processor.process_masks(masks)
                # ID 재부여
                for lane in skeleton_lanes:
                    lane["id"] = len(lanes)
                    lanes.append(lane)
            else:
                # 기존 방식: 마스크에서 직접 폴리라인 추출
                for i, mask in enumerate(masks):
                    points = self._mask_to_polyline(mask, pixel_interval, poly_degree=poly_degree)

                    if len(points) >= 2:
                        lanes.append({
                            "id": len(lanes),
                            "points": points,
                            "mask": (mask * 255).astype(np.uint8)
                        })

        return lanes

    def _mask_to_polyline(
        self,
        mask: np.ndarray,
        pixel_interval: int = 20,
        min_width: int = 3,
        max_x_delta: int = 50,
        poly_degree: int = 2
    ) -> List[List[int]]:
        """
        세그먼테이션 마스크를 폴리라인으로 변환.
        차선은 세로로 길쭉하므로 y좌표 기준으로 샘플링.

        Args:
            mask: 세그먼테이션 마스크
            pixel_interval: 샘플링 간격 (픽셀 단위, 기본 20px)
            min_width: 최소 마스크 두께 (이하는 노이즈로 판단, 기본 3px)
            max_x_delta: 최대 x 변화량 (초과 시 이상치로 판단, 기본 50px)
            poly_degree: 다항식 피팅 차수 (0이면 피팅 안함, 기본 2차)
        """
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 세로 방향으로 픽셀 간격 기준 샘플링
        raw_points = []
        y_values = range(y, y + h, pixel_interval)

        for yi in y_values:
            row = mask_uint8[yi, :]
            x_indices = np.where(row > 0)[0]

            # 마스크 두께 필터링: 너무 얇으면 신뢰도 낮음
            if len(x_indices) >= min_width:
                x_center = int(np.mean(x_indices))
                raw_points.append([x_center, int(yi)])

        # 이상치 제거: x 변화가 급격한 포인트 필터링
        points = self._filter_outliers(raw_points, max_x_delta)

        # 다항식 피팅으로 스무딩 (지그재그 제거)
        if poly_degree > 0 and len(points) >= poly_degree + 1:
            points = self._smooth_with_polynomial(points, poly_degree)

        return points

    def _filter_outliers(self, points: List[List[int]], max_x_delta: int) -> List[List[int]]:
        """
        연속된 포인트 간 x 변화가 급격한 이상치 제거.

        Args:
            points: 원본 포인트 리스트
            max_x_delta: 허용 최대 x 변화량
        """
        if len(points) < 2:
            return points

        filtered = [points[0]]

        for i in range(1, len(points)):
            prev_x = filtered[-1][0]
            curr_x = points[i][0]

            # 이전 포인트와 x 차이가 허용 범위 내인 경우만 추가
            if abs(curr_x - prev_x) <= max_x_delta:
                filtered.append(points[i])

        return filtered

    def _smooth_with_polynomial(
        self,
        points: List[List[int]],
        degree: int = 2
    ) -> List[List[int]]:
        """
        다항식 피팅으로 포인트를 스무딩.
        차선은 세로로 길기 때문에 y를 독립변수로, x를 종속변수로 피팅.

        Args:
            points: 원본 포인트 리스트 [[x, y], ...]
            degree: 다항식 차수 (기본 2차, 곡선 도로는 3차 권장)

        Returns:
            스무딩된 포인트 리스트
        """
        if len(points) < degree + 1:
            # 피팅에 필요한 최소 포인트 수 미달
            return points

        # y를 독립변수, x를 종속변수로 분리
        y_coords = np.array([p[1] for p in points])
        x_coords = np.array([p[0] for p in points])

        try:
            # 다항식 피팅: x = f(y)
            coeffs = np.polyfit(y_coords, x_coords, degree)
            poly = np.poly1d(coeffs)

            # 피팅된 다항식으로 새로운 x 좌표 계산
            smoothed_x = poly(y_coords)

            # 정수로 변환하여 반환
            smoothed_points = [
                [int(round(x)), int(y)]
                for x, y in zip(smoothed_x, y_coords)
            ]

            return smoothed_points

        except np.RankWarning:
            # 피팅 실패 시 원본 반환
            print("[SAM3] 다항식 피팅 경고: 원본 포인트 유지")
            return points
        except Exception as e:
            print(f"[SAM3] 다항식 피팅 오류: {e}")
            return points


def detect_lanes_sam3(
    image_path: str,
    model_path: str = "models/sam3.pt",
    text_prompts: List[str] = None,
    device: str = "cuda",
    conf: float = 0.5,
    poly_degree: int = 2,
    use_skeleton: bool = False,
    skeleton_config: Optional[SkeletonProcessorConfig] = None
) -> List[Dict]:
    """
    SAM3로 차선 검출하는 헬퍼 함수

    Args:
        image_path: 이미지 파일 경로
        model_path: SAM3 모델 경로
        text_prompts: 텍스트 프롬프트
        device: cuda 또는 cpu
        conf: 신뢰도 임계값
        poly_degree: 다항식 피팅 차수 (0이면 피팅 안함, 기본 2차, 곡선 도로는 3차 권장)
        use_skeleton: 스켈레톤 기반 후처리 사용 여부 (기본 False)
        skeleton_config: 스켈레톤 프로세서 설정
    """
    detector = LaneDetectorSAM3(
        model_path=model_path,
        device=device,
        conf=conf,
        use_skeleton=use_skeleton,
        skeleton_config=skeleton_config
    )
    return detector.detect_lanes(image_path, text_prompts, poly_degree=poly_degree)
