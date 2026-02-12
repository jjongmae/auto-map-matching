"""
스켈레톤 기반 차선 후처리 모듈 (단순화 버전)

SAM3 마스크를 스켈레톤화하여 차선 중앙점으로 변환합니다.
파이프라인:
1. 마스크 → 스켈레톤 (skimage.morphology.skeletonize)
2. 스켈레톤 픽셀 → y 정렬 → 구간별 x 중앙값
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import cv2


@dataclass
class SkeletonProcessorConfig:
    """스켈레톤 프로세서 설정"""
    mask_threshold: float = 0.5       # 마스크 이진화 임계값
    mask_iou_threshold: float = 0.5   # 마스크 중복 판정 IoU 임계값
    mask_contain_threshold: float = 0.7  # 마스크 포함 비율 임계값
    output_point_interval: float = 40.0  # 출력 점 간격 (px)


class SkeletonLaneProcessor:
    """스켈레톤 기반 차선 후처리 프로세서 (단순화 버전)"""

    def __init__(self, config: Optional[SkeletonProcessorConfig] = None):
        """
        Args:
            config: 프로세서 설정. None이면 기본값 사용
        """
        self.config = config or SkeletonProcessorConfig()
        self._skeletonize = None  # 지연 로딩

    def _ensure_skeletonize(self):
        """skimage.morphology 함수 지연 로딩"""
        if self._skeletonize is None:
            from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
            self._skeletonize = skeletonize
            self._remove_small_objects = remove_small_objects
            self._binary_closing = binary_closing
            self._disk = disk

    def process_masks(self, masks: List[np.ndarray]) -> tuple:
        """
        마스크 리스트를 처리하여 점 목록과 합성 마스크 반환

        Args:
            masks: 마스크 배열 리스트 (각각 2D numpy array)

        Returns:
            (points, combined_mask): 점 목록 [[x, y], ...], 필터링된 마스크 합성본
        """
        print(f"[스켈레톤] 입력 마스크: {len(masks)}개")

        # 마스크 중복 제거 (NMS)
        filtered_masks = self._remove_duplicate_masks(masks)
        print(f"[스켈레톤] 중복 제거 후: {len(filtered_masks)}개")

        all_points = []

        for i, mask in enumerate(filtered_masks):
            # 디버그: 마스크 bounding box 출력
            ys, xs = np.where(mask > 0.5)
            if len(ys) > 0:
                bbox_str = f"x:[{xs.min()}-{xs.max()}] y:[{ys.min()}-{ys.max()}]"
            else:
                bbox_str = "empty"

            points = self.process_single_mask(mask, mask_id=i)
            if points:
                all_points.extend(points)
                print(f"[스켈레톤] 마스크 #{i}: {len(points)}개 점 추출, {bbox_str}, 점={points}")

        print(f"[스켈레톤] 총 점 개수 (필터링 전): {len(all_points)}개")

        # 필터링된 마스크들만 합성
        if len(filtered_masks) > 0:
            combined_mask = np.zeros_like(filtered_masks[0], dtype=np.uint8)
            for mask in filtered_masks:
                combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8))
            # 개별 마스크도 uint8로 변환하여 반환
            individual_masks = [(mask * 255).astype(np.uint8) for mask in filtered_masks]

            # 전체 점에 대한 통합 중복 제거 (Distance Transform 기반)
            if len(all_points) > 1:
                # 합성 마스크에 대한 Distance Transform 계산
                combined_binary = (combined_mask > 127).astype(np.uint8)
                global_dist_map = cv2.distanceTransform(combined_binary, cv2.DIST_L2, 5)

                # 전체 점에 대해 인접 점 필터링 적용
                all_points = self._filter_points_globally(all_points, global_dist_map, min_dist=15.0)
                print(f"[스켈레톤] 총 점 개수 (전역 필터링 후): {len(all_points)}개")
        else:
            combined_mask = None
            individual_masks = []

        return all_points, combined_mask, individual_masks

    def _compute_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> tuple:
        """두 마스크 간 IoU와 포함 비율 계산

        Returns:
            (iou, contain_ratio): IoU와 작은 마스크의 포함 비율
        """
        m1 = mask1.astype(bool)
        m2 = mask2.astype(bool)

        intersection = np.logical_and(m1, m2).sum()
        union = np.logical_or(m1, m2).sum()

        if union == 0:
            return 0.0, 0.0

        iou = intersection / union

        # 작은 마스크가 큰 마스크에 포함된 비율
        smaller_area = min(m1.sum(), m2.sum())
        if smaller_area == 0:
            contain_ratio = 0.0
        else:
            contain_ratio = intersection / smaller_area

        return iou, contain_ratio

    def _remove_duplicate_masks(self, masks: List[np.ndarray]) -> List[np.ndarray]:
        """IoU 및 포함 비율 기반 중복 마스크 제거 (NMS)"""
        if len(masks) <= 1:
            return masks

        # 마스크 면적 계산 (큰 것 우선)
        areas = [m.astype(bool).sum() for m in masks]
        sorted_indices = np.argsort(areas)[::-1]

        keep = []
        suppressed = set()

        for i in sorted_indices:
            if i in suppressed:
                continue

            keep.append(i)

            for j in sorted_indices:
                if j in suppressed or j == i:
                    continue

                iou, contain_ratio = self._compute_mask_overlap(masks[i], masks[j])

                if iou > self.config.mask_iou_threshold or contain_ratio > self.config.mask_contain_threshold:
                    suppressed.add(j)

        keep_set = set(keep)
        return [masks[i] for i in range(len(masks)) if i in keep_set]

    def process_single_mask(self, mask: np.ndarray, mask_id: int = -1) -> List[List[int]]:
        """
        단일 마스크 처리: 전처리 -> 스켈레톤 -> 축 기반 동적 샘플링 -> 중복 점 필터링
        """
        self._ensure_skeletonize()

        # 1. 마스크 이진화
        binary_mask = self._binarize_mask(mask).astype(bool)
        px_before = np.sum(binary_mask)
        
        # 2. 전처리: 노이즈 제거 및 형태 보정
        try:
            binary_mask = self._remove_small_objects(binary_mask, min_size=50)
            binary_mask = self._binary_closing(binary_mask, self._disk(3))
        except Exception as e:
            # print(f"[스켈레톤 #{mask_id}] 전처리 오류: {e}")
            pass

        px_after = np.sum(binary_mask)

        # 3. 스켈레톤 변환
        skeleton = self._skeletonize(binary_mask).astype(np.uint8)

        # 스켈레톤 점 추출
        y_coords, x_coords = np.where(skeleton > 0)
        px_skeleton = len(y_coords)

        if px_skeleton == 0:
            return []

        # 4. 거리 변환 (Distance Transform) 계산 - 차선 중심일수록 값이 큼
        # binary_mask는 bool 타입이므로 uint8로 변환 필요
        dist_transform = cv2.distanceTransform(binary_mask.astype(np.uint8), cv2.DIST_L2, 5)

        # 범위 및 길이 계산
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        
        y_range = y_max - y_min
        x_range = x_max - x_min
        euclidean_dist = np.sqrt(x_range**2 + y_range**2)
        
        if euclidean_dist == 0:
            return []

        base_interval = self.config.output_point_interval
        points = []

        # --- A. 세로형 차선 (Height >= Width) ---
        if y_range >= x_range:
            # 기울기 보정: 실제 거리가 base_interval이 되도록 스캔 간격(y_step) 조정
            if euclidean_dist > 0:
                y_step = max(int(base_interval * (y_range / euclidean_dist)), 10) 
            else:
                y_step = int(base_interval)

            # 1. 시작점 (Top)
            start_indices = np.where(y_coords == y_min)[0]
            if len(start_indices) > 0:
                x_start = int(np.median(x_coords[start_indices]))
                points.append([x_start, int(y_min)])

            # 2. 중간 점 (Y축 스캔)
            for y_target in range(y_min + y_step, y_max, y_step):
                search_radius = y_step // 2
                search_min = max(y_target - search_radius, y_min)
                search_max = min(y_target + search_radius, y_max)
                
                mask_in_range = (y_coords >= search_min) & (y_coords < search_max)
                if mask_in_range.any():
                    xs = x_coords[mask_in_range]
                    ys = y_coords[mask_in_range]
                    
                    x_med = np.median(xs)
                    y_med = np.median(ys)
                    
                    # Nearest Neighbor
                    dists = (xs - x_med)**2 + (ys - y_med)**2
                    min_idx = np.argmin(dists)
                    
                    points.append([int(xs[min_idx]), int(ys[min_idx])])

            # 3. 끝점 (Bottom)
            end_indices = np.where(y_coords == y_max)[0]
            if len(end_indices) > 0:
                x_end = int(np.median(x_coords[end_indices]))
                y_end = int(y_max)
                
                if not points:
                    points.append([x_end, y_end])
                else:
                    _, last_y = points[-1]
                    if abs(y_end - last_y) > 5:
                        points.append([x_end, y_end])
                    else:
                        points[-1] = [x_end, y_end]

        # --- B. 가로형/대각선 차선 (Width > Height) ---
        else:
            # 기울기 보정 (X축 기준)
            if euclidean_dist > 0:
                x_step = max(int(base_interval * (x_range / euclidean_dist)), 10)
            else:
                x_step = int(base_interval)

            # 1. 시작점 (Left)
            start_indices = np.where(x_coords == x_min)[0]
            if len(start_indices) > 0:
                y_start = int(np.median(y_coords[start_indices]))
                points.append([int(x_min), y_start])

            # 2. 중간 점 (X축 스캔)
            for x_target in range(x_min + x_step, x_max, x_step):
                search_radius = x_step // 2
                search_min = max(x_target - search_radius, x_min)
                search_max = min(x_target + search_radius, x_max)
                
                mask_in_range = (x_coords >= search_min) & (x_coords < search_max)
                if mask_in_range.any():
                    xs = x_coords[mask_in_range]
                    ys = y_coords[mask_in_range]
                    
                    x_med = np.median(xs)
                    y_med = np.median(ys)
                    
                    # Nearest Neighbor
                    dists = (xs - x_med)**2 + (ys - y_med)**2
                    min_idx = np.argmin(dists)
                    
                    points.append([int(xs[min_idx]), int(ys[min_idx])])

            # 3. 끝점 (Right)
            end_indices = np.where(x_coords == x_max)[0]
            if len(end_indices) > 0:
                y_end = int(np.median(y_coords[end_indices]))
                x_end = int(x_max)
                
                if not points:
                    points.append([x_end, y_end])
                else:
                    last_x, _ = points[-1]
                    if abs(x_end - last_x) > 5:
                        points.append([x_end, y_end])
                    else:
                        points[-1] = [x_end, y_end]

        return points

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 이진화"""
        if mask.dtype == bool:
            return mask.astype(np.uint8)
        elif mask.dtype == np.float32 or mask.dtype == np.float64:
            return (mask > self.config.mask_threshold).astype(np.uint8)
        else:
            threshold = int(self.config.mask_threshold * 255)
            return (mask > threshold).astype(np.uint8)

    def _filter_points_globally(self, points: List[List[int]], dist_map: np.ndarray, min_dist: float = 15.0) -> List[List[int]]:
        """
        전역 중복 점 필터링 (모든 마스크에서 추출된 점 대상):
        - 모든 점을 Distance Transform 점수 기준으로 정렬
        - 점수 높은 점부터 선택, 가까운 점(min_dist 이내)은 제거

        Args:
            points: 모든 마스크에서 추출된 점 리스트 [[x, y], ...]
            dist_map: 합성 마스크의 Distance Transform 맵
            min_dist: 최소 거리 임계값 (이 거리 이내의 점은 중복으로 판단)

        Returns:
            필터링된 점 리스트
        """
        if len(points) <= 1:
            return points

        h, w = dist_map.shape

        # 각 점에 대해 Distance Transform 점수 계산
        scored_points = []
        for p in points:
            x, y = p
            # 범위 체크 후 점수 추출
            if 0 <= x < w and 0 <= y < h:
                score = dist_map[y, x]
            else:
                score = 0.0
            scored_points.append({'point': p, 'score': score})

        # 점수 높은 순으로 정렬 (차선 중심에 가까운 점 우선)
        scored_points.sort(key=lambda x: x['score'], reverse=True)

        # Greedy 방식으로 점 선택
        accepted = []
        min_dist_sq = min_dist ** 2  # 제곱 거리로 비교 (sqrt 연산 회피)

        for curr in scored_points:
            curr_p = curr['point']
            cx, cy = curr_p

            # 이미 선택된 점들과 거리 비교
            is_valid = True
            for acc in accepted:
                ax, ay = acc['point']
                dist_sq = (cx - ax) ** 2 + (cy - ay) ** 2

                if dist_sq < min_dist_sq:
                    # 가까운 점이 이미 있으면 (점수가 더 높은 점), 현재 점 제외
                    is_valid = False
                    break

            if is_valid:
                accepted.append(curr)

        # 결과 점 리스트 추출 (정렬 없이 반환 - 호출측에서 필요시 정렬)
        return [item['point'] for item in accepted]
