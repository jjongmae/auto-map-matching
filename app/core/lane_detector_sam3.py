"""
SAM3를 사용한 차선 검출 모듈

Ultralytics SAM3 모델을 사용하여 이미지에서 차선을 세그먼테이션합니다.
참고: https://docs.ultralytics.com/models/sam-3/
"""

import cv2
import numpy as np
from typing import List, Dict, Optional


class LaneDetectorSAM3:
    """SAM3 기반 차선 검출기"""

    def __init__(self, model_path: str = "models/sam3.pt", device: str = "cuda"):
        """
        Args:
            model_path: SAM3 모델 파일 경로
            device: cuda 또는 cpu
        """
        self.model_path = model_path
        self.device = device
        self.predictor = None
        self.model_loaded = False

    def load_model(self) -> bool:
        """SAM3 모델 로드"""
        try:
            from ultralytics.models.sam import SAM3SemanticPredictor

            overrides = dict(
                conf=0.25,
                task="segment",
                mode="predict",
                model=self.model_path,
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
        pixel_interval: int = 20
    ) -> List[Dict]:
        """
        이미지에서 차선을 검출합니다.

        Args:
            image_path: 이미지 파일 경로
            text_prompts: 텍스트 프롬프트 (기본: ["lane", "road marking"])
            pixel_interval: 점 샘플링 간격 (픽셀 단위, 기본 20px)

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

            lanes = self._process_results(results, pixel_interval)
            print(f"[SAM3] {len(lanes)}개의 차선 검출됨")
            return lanes

        except Exception as e:
            print(f"[SAM3] 검출 오류: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _process_results(self, results, pixel_interval: int) -> List[Dict]:
        """SAM3 결과를 차선 데이터로 변환"""
        lanes = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                # 마스크에서 폴리라인 추출
                points = self._mask_to_polyline(mask, pixel_interval)

                if len(points) >= 2:
                    lanes.append({
                        "id": len(lanes),
                        "points": points,
                        "mask": (mask * 255).astype(np.uint8)
                    })

        return lanes

    def _mask_to_polyline(self, mask: np.ndarray, pixel_interval: int = 20) -> List[List[int]]:
        """
        세그먼테이션 마스크를 폴리라인으로 변환.
        차선은 세로로 길쭉하므로 y좌표 기준으로 샘플링.

        Args:
            mask: 세그먼테이션 마스크
            pixel_interval: 샘플링 간격 (픽셀 단위, 기본 20px)
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
        points = []
        y_values = range(y, y + h, pixel_interval)

        for yi in y_values:
            row = mask_uint8[yi, :]
            x_indices = np.where(row > 0)[0]

            if len(x_indices) > 0:
                x_center = int(np.mean(x_indices))
                points.append([x_center, int(yi)])

        return points


def detect_lanes_sam3(
    image_path: str,
    model_path: str = "models/sam3.pt",
    text_prompts: List[str] = None,
    device: str = "cuda"
) -> List[Dict]:
    """SAM3로 차선 검출하는 헬퍼 함수"""
    detector = LaneDetectorSAM3(model_path=model_path, device=device)
    return detector.detect_lanes(image_path, text_prompts)
