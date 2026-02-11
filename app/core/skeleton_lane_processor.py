"""
스켈레톤 기반 차선 후처리 모듈

SAM3 마스크를 스켈레톤화하여 정교한 폴리라인으로 변환합니다.
파이프라인:
1. 마스크 → 스켈레톤 (skimage.morphology.skeletonize)
2. 스켈레톤 → 점 추출 + KDTree로 근접점 병합
3. 연결성 기반 클러스터링 + 곡률 연속성 검증
4. B-스플라인 피팅 (scipy.interpolate.splprep/splev)
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import deque

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev


@dataclass
class SkeletonProcessorConfig:
    """스켈레톤 프로세서 설정"""
    mask_threshold: float = 0.5       # 마스크 이진화 임계값
    mask_iou_threshold: float = 0.5   # 마스크 중복 판정 IoU 임계값
    mask_contain_threshold: float = 0.7  # 마스크 포함 비율 임계값
    merge_radius: float = 10.0        # 근접점 병합 반경 (px)
    neighbor_radius: float = 12.0     # 연결 판정 반경 (px)
    min_cluster_points: int = 3       # 최소 클러스터 점 수
    max_curvature_change: float = 1.0 # 최대 곡률 변화 (rad)
    polyline_distance_threshold: float = 15.0  # 폴리라인 중복 판정 최소 거리 (px)
    polyline_overlap_ratio: float = 0.5       # 폴리라인 중복 판정 겹침 비율
    spline_smoothing: float = 5.0     # 스플라인 평활화 계수 (노이즈 제거용)
    spline_degree: int = 3            # 스플라인 차수
    output_point_interval: float = 40.0  # 출력 점 간격 (px)


class SkeletonLaneProcessor:
    """스켈레톤 기반 차선 후처리 프로세서"""

    def __init__(self, config: Optional[SkeletonProcessorConfig] = None):
        """
        Args:
            config: 프로세서 설정. None이면 기본값 사용
        """
        self.config = config or SkeletonProcessorConfig()
        self._skeletonize = None  # 지연 로딩

    def _ensure_skeletonize(self):
        """skimage.morphology.skeletonize 함수 지연 로딩"""
        if self._skeletonize is None:
            from skimage.morphology import skeletonize
            self._skeletonize = skeletonize

    def process_masks(self, masks: List[np.ndarray]) -> List[Dict]:
        """
        마스크 리스트를 처리하여 차선 데이터로 변환

        Args:
            masks: 마스크 배열 리스트 (각각 2D numpy array)

        Returns:
            차선 데이터 리스트 [{"id": int, "points": [[x,y],...], "mask": np.ndarray}, ...]
        """
        # 마스크 중복 제거 (NMS)
        filtered_masks = self._remove_duplicate_masks(masks)

        lanes = []

        for i, mask in enumerate(filtered_masks):
            result = self.process_single_mask(mask)

            if result is not None:
                for points in result:
                    if len(points) >= 2:
                        lanes.append({
                            "id": len(lanes),
                            "points": points,
                            "mask": (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask.astype(np.uint8)
                        })

        # 폴리라인 중복 제거
        filtered_lanes = self._remove_duplicate_polylines(lanes)

        # ID 재부여
        for i, lane in enumerate(filtered_lanes):
            lane["id"] = i

        return filtered_lanes

    def _compute_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> Tuple[float, float]:
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
        # 면적 기준 내림차순 정렬
        sorted_indices = np.argsort(areas)[::-1]

        keep = []
        suppressed = set()

        for i in sorted_indices:
            if i in suppressed:
                continue

            keep.append(i)

            # 나머지 마스크와 비교
            for j in sorted_indices:
                if j in suppressed or j == i:
                    continue

                iou, contain_ratio = self._compute_mask_overlap(masks[i], masks[j])

                # IoU가 높거나, 작은 마스크가 큰 마스크에 많이 포함되면 중복으로 판정
                if iou > self.config.mask_iou_threshold or contain_ratio > self.config.mask_contain_threshold:
                    suppressed.add(j)

        # 원래 순서 유지하면서 필터링
        keep_set = set(keep)
        return [masks[i] for i in range(len(masks)) if i in keep_set]

    def _compute_polyline_overlap(self, points1: List[List[int]], points2: List[List[int]]) -> Tuple[float, float]:
        """두 폴리라인 간 최소 거리와 겹침 비율 계산

        Returns:
            (min_distance, overlap_ratio): 최소 거리와 짧은 폴리라인의 겹침 비율
        """
        if not points1 or not points2:
            return float('inf'), 0.0

        p1 = np.array(points1)
        p2 = np.array(points2)
        threshold = self.config.polyline_distance_threshold

        # 최소 거리 계산
        min_distance = float('inf')
        for pt in p1:
            dists = np.linalg.norm(p2 - pt, axis=1)
            min_distance = min(min_distance, np.min(dists))

        # 겹침 비율 계산 (짧은 폴리라인 기준)
        # p1의 각 점 중 p2와 가까운 점의 비율
        close_count_1 = 0
        for pt in p1:
            dists = np.linalg.norm(p2 - pt, axis=1)
            if np.min(dists) < threshold:
                close_count_1 += 1

        # p2의 각 점 중 p1과 가까운 점의 비율
        close_count_2 = 0
        for pt in p2:
            dists = np.linalg.norm(p1 - pt, axis=1)
            if np.min(dists) < threshold:
                close_count_2 += 1

        # 짧은 폴리라인의 겹침 비율
        overlap_ratio_1 = close_count_1 / len(p1) if len(p1) > 0 else 0
        overlap_ratio_2 = close_count_2 / len(p2) if len(p2) > 0 else 0
        overlap_ratio = max(overlap_ratio_1, overlap_ratio_2)

        return min_distance, overlap_ratio

    def _remove_duplicate_polylines(self, lanes: List[Dict]) -> List[Dict]:
        """최소 거리 + 겹침 비율 기반 중복 폴리라인 제거"""
        if len(lanes) <= 1:
            return lanes

        # 폴리라인 길이 계산 (긴 것 우선)
        def polyline_length(points):
            if len(points) < 2:
                return 0
            pts = np.array(points)
            return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))

        lengths = [polyline_length(lane["points"]) for lane in lanes]
        sorted_indices = np.argsort(lengths)[::-1]

        keep = []
        suppressed = set()

        for i in sorted_indices:
            if i in suppressed:
                continue

            keep.append(i)

            for j in sorted_indices:
                if j in suppressed or j == i:
                    continue

                min_dist, overlap_ratio = self._compute_polyline_overlap(
                    lanes[i]["points"], lanes[j]["points"]
                )

                # 최소 거리가 가깝고 AND 겹침 비율이 높으면 중복으로 판정
                if min_dist < self.config.polyline_distance_threshold and \
                   overlap_ratio > self.config.polyline_overlap_ratio:
                    suppressed.add(j)

        # 원래 순서 유지
        keep_set = set(keep)
        return [lanes[i] for i in range(len(lanes)) if i in keep_set]

    def process_single_mask(self, mask: np.ndarray) -> Optional[List[List[List[int]]]]:
        """
        단일 마스크를 처리하여 폴리라인 점들로 변환

        Args:
            mask: 2D 마스크 배열

        Returns:
            폴리라인 리스트 [[[x, y], ...], ...] 또는 점이 없으면 None
        """
        self._ensure_skeletonize()

        # Stage 1: 마스크 → 스켈레톤
        binary_mask = self._binarize_mask(mask)
        skeleton = self._extract_skeleton(binary_mask)

        # Stage 2: 스켈레톤 → 점 추출 + 근접점 병합
        points = self._skeleton_to_points(skeleton)
        if len(points) < self.config.min_cluster_points:
            return None

        merged_points = self._merge_nearby_points(points)
        if len(merged_points) < self.config.min_cluster_points:
            return None

        # Stage 3: 연결성 기반 클러스터링
        graph = self._build_connectivity_graph(merged_points)
        clusters = self._find_connected_components(graph, len(merged_points))

        # 클러스터별 처리
        result_polylines = []
        for cluster_indices in clusters:
            if len(cluster_indices) < self.config.min_cluster_points:
                continue

            cluster_points = merged_points[cluster_indices]

            # 점 순서 정렬
            ordered_points = self._order_cluster_points(cluster_points)

            # 곡률 연속성 검증 및 분리
            segments = self._validate_curvature_continuity(ordered_points)

            for segment in segments:
                if len(segment) < self.config.min_cluster_points:
                    continue

                # Stage 4: B-스플라인 피팅 및 리샘플링
                try:
                    fitted_points = self._fit_and_resample_spline(segment)
                    if len(fitted_points) >= 2:
                        result_polylines.append(fitted_points)
                except Exception:
                    # 스플라인 피팅 실패 시 원본 점 사용
                    resampled = self._resample_points(segment)
                    if len(resampled) >= 2:
                        result_polylines.append(resampled)

        return result_polylines if result_polylines else None

    def _binarize_mask(self, mask: np.ndarray) -> np.ndarray:
        """마스크 이진화"""
        if mask.dtype == bool:
            # bool 타입은 그대로 uint8로 변환
            return mask.astype(np.uint8)
        elif mask.dtype == np.float32 or mask.dtype == np.float64:
            return (mask > self.config.mask_threshold).astype(np.uint8)
        else:
            # uint8 마스크의 경우
            threshold = int(self.config.mask_threshold * 255)
            return (mask > threshold).astype(np.uint8)

    def _extract_skeleton(self, binary_mask: np.ndarray) -> np.ndarray:
        """skeletonize로 스켈레톤 추출"""
        # skimage.morphology.skeletonize는 boolean 입력을 기대
        skeleton = self._skeletonize(binary_mask.astype(bool))
        return skeleton.astype(np.uint8)

    def _skeleton_to_points(self, skeleton: np.ndarray) -> np.ndarray:
        """스켈레톤에서 점 좌표 추출 (y, x → x, y 변환)"""
        # np.where는 (row, col) = (y, x) 순서로 반환
        y_coords, x_coords = np.where(skeleton > 0)
        # (x, y) 형식으로 변환
        points = np.column_stack([x_coords, y_coords])
        return points.astype(np.float64)

    def _merge_nearby_points(self, points: np.ndarray) -> np.ndarray:
        """cKDTree로 근접점 병합"""
        if len(points) == 0:
            return points

        tree = cKDTree(points)
        merge_radius = self.config.merge_radius

        # 방문 여부 추적
        visited = np.zeros(len(points), dtype=bool)
        merged = []

        for i in range(len(points)):
            if visited[i]:
                continue

            # 반경 내 모든 점 찾기
            indices = tree.query_ball_point(points[i], merge_radius)

            # 이미 방문한 점 제외
            unvisited_indices = [idx for idx in indices if not visited[idx]]

            if unvisited_indices:
                # 평균 위치 계산
                cluster_points = points[unvisited_indices]
                centroid = cluster_points.mean(axis=0)
                merged.append(centroid)

                # 모두 방문 처리
                for idx in unvisited_indices:
                    visited[idx] = True

        return np.array(merged) if merged else np.array([]).reshape(0, 2)

    def _build_connectivity_graph(self, points: np.ndarray) -> Dict[int, List[int]]:
        """연결 그래프 구성"""
        if len(points) == 0:
            return {}

        tree = cKDTree(points)
        neighbor_radius = self.config.neighbor_radius

        graph = {i: [] for i in range(len(points))}

        for i in range(len(points)):
            # 반경 내 이웃 찾기 (자기 자신 제외)
            indices = tree.query_ball_point(points[i], neighbor_radius)
            neighbors = [idx for idx in indices if idx != i]
            graph[i] = neighbors

        return graph

    def _find_connected_components(self, graph: Dict[int, List[int]], n_points: int) -> List[List[int]]:
        """BFS로 연결 컴포넌트 분리"""
        visited = set()
        components = []

        for start in range(n_points):
            if start in visited:
                continue

            # BFS
            component = []
            queue = deque([start])

            while queue:
                node = queue.popleft()
                if node in visited:
                    continue

                visited.add(node)
                component.append(node)

                for neighbor in graph.get(node, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            if component:
                components.append(component)

        return components

    def _order_cluster_points(self, points: np.ndarray) -> np.ndarray:
        """PCA 기반 점 순서 정렬

        점들이 가장 많이 퍼진 방향(주축)을 찾아서 그 방향으로 정렬합니다.
        지그재그 문제를 방지하고 차선의 자연스러운 흐름을 유지합니다.
        """
        if len(points) <= 2:
            return points

        # 1. 점들의 중심 계산
        centroid = points.mean(axis=0)
        centered = points - centroid

        # 2. 공분산 행렬로 주축 방향 계산 (PCA)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 가장 큰 고유값에 해당하는 고유벡터 = 주축 방향
        principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

        # 3. 각 점을 주축에 투영
        projections = np.dot(centered, principal_axis)

        # 4. 투영값 순서로 정렬 (작은 값 → 큰 값)
        sorted_indices = np.argsort(projections)

        # 5. 차선이 위→아래로 진행하도록 방향 조정
        # (첫 점의 y가 마지막 점의 y보다 크면 순서 뒤집기)
        ordered_points = points[sorted_indices]
        if ordered_points[0, 1] > ordered_points[-1, 1]:
            ordered_points = ordered_points[::-1]

        # 6. 이동평균으로 지그재그 제거
        smoothed_points = self._smooth_zigzag(ordered_points)

        return smoothed_points

    def _smooth_zigzag(self, points: np.ndarray, window: int = 3) -> np.ndarray:
        """이동평균으로 지그재그 제거

        Args:
            points: 정렬된 점 배열
            window: 이동평균 윈도우 크기 (홀수 권장)

        Returns:
            스무딩된 점 배열
        """
        if len(points) <= window:
            return points

        smoothed = np.copy(points).astype(np.float64)
        half = window // 2

        # 양 끝점은 유지하고 중간 점들만 스무딩
        for i in range(half, len(points) - half):
            smoothed[i] = points[i - half:i + half + 1].mean(axis=0)

        return smoothed

    def _compute_curvature(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """3점 원 공식으로 곡률 계산 (Menger curvature)"""
        # 세 점으로 이루는 삼각형의 면적의 4배 / (세 변의 길이의 곱)
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)

        # 작은 값 방지
        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            return 0.0

        # 삼각형 면적 (외적의 절반)
        cross = np.cross(p2 - p1, p3 - p1)
        area = abs(cross) / 2.0

        # 곡률 = 4 * 면적 / (a * b * c)
        curvature = 4.0 * area / (a * b * c)

        return curvature

    def _validate_curvature_continuity(self, points: np.ndarray) -> List[np.ndarray]:
        """곡률 급변점에서 분리"""
        if len(points) < 4:
            return [points]

        segments = []
        current_segment_start = 0
        max_curvature_change = self.config.max_curvature_change

        # 곡률 계산
        curvatures = []
        for i in range(1, len(points) - 1):
            k = self._compute_curvature(points[i-1], points[i], points[i+1])
            curvatures.append(k)

        # 곡률 변화가 급격한 지점 찾기
        for i in range(1, len(curvatures)):
            curvature_change = abs(curvatures[i] - curvatures[i-1])

            if curvature_change > max_curvature_change:
                # 분리점 발견
                segment = points[current_segment_start:i+2]  # i+1까지 포함
                if len(segment) >= self.config.min_cluster_points:
                    segments.append(segment)
                current_segment_start = i + 1

        # 마지막 세그먼트
        if current_segment_start < len(points):
            segment = points[current_segment_start:]
            if len(segment) >= self.config.min_cluster_points:
                segments.append(segment)

        return segments if segments else [points]

    def _fit_spline(self, points: np.ndarray) -> Tuple:
        """splprep으로 B-스플라인 피팅"""
        # 점이 너무 적으면 낮은 차수 사용
        k = min(self.config.spline_degree, len(points) - 1)
        if k < 1:
            raise ValueError(f"스플라인 피팅에 점이 부족합니다: {len(points)}개")

        x = points[:, 0]
        y = points[:, 1]

        # splprep 파라미터
        s = self.config.spline_smoothing

        try:
            tck, u = splprep([x, y], k=k, s=s)
            return tck, u
        except Exception as e:
            # 피팅 실패 시 더 낮은 차수로 재시도
            if k > 1:
                tck, u = splprep([x, y], k=1, s=s)
                return tck, u
            raise e

    def _remove_duplicate_points(self, points: List[List[int]]) -> List[List[int]]:
        """연속된 중복 좌표 제거"""
        if len(points) <= 1:
            return points

        result = [points[0]]
        for p in points[1:]:
            if p[0] != result[-1][0] or p[1] != result[-1][1]:
                result.append(p)

        return result

    def _resample_spline(self, tck: Tuple, total_length: float) -> List[List[int]]:
        """splev로 균등 간격 리샘플링"""
        interval = self.config.output_point_interval
        n_points = max(2, int(total_length / interval) + 1)

        u_new = np.linspace(0, 1, n_points)
        x_new, y_new = splev(u_new, tck)

        # 정수 좌표로 변환
        points = [[int(round(x)), int(round(y))] for x, y in zip(x_new, y_new)]

        # 중복 제거
        return self._remove_duplicate_points(points)

    def _fit_and_resample_spline(self, points: np.ndarray) -> List[List[int]]:
        """스플라인 피팅 후 리샘플링"""
        # 곡선 길이 추정
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_length = np.sum(segment_lengths)

        if total_length < self.config.output_point_interval:
            # 길이가 너무 짧으면 시작점과 끝점만 반환
            return [
                [int(round(points[0][0])), int(round(points[0][1]))],
                [int(round(points[-1][0])), int(round(points[-1][1]))]
            ]

        # 스플라인 피팅
        tck, u = self._fit_spline(points)

        # 균등 간격 리샘플링
        return self._resample_spline(tck, total_length)

    def _resample_points(self, points: np.ndarray) -> List[List[int]]:
        """스플라인 없이 균등 간격 리샘플링 (폴백용)"""
        if len(points) < 2:
            return [[int(round(p[0])), int(round(p[1]))] for p in points]

        # 누적 거리 계산
        diffs = np.diff(points, axis=0)
        segment_lengths = np.linalg.norm(diffs, axis=1)
        cumulative_lengths = np.concatenate([[0], np.cumsum(segment_lengths)])
        total_length = cumulative_lengths[-1]

        if total_length < self.config.output_point_interval:
            return [
                [int(round(points[0][0])), int(round(points[0][1]))],
                [int(round(points[-1][0])), int(round(points[-1][1]))]
            ]

        # 등간격 샘플링 위치 계산
        interval = self.config.output_point_interval
        n_samples = max(2, int(total_length / interval) + 1)
        target_distances = np.linspace(0, total_length, n_samples)

        # 보간
        resampled = []
        for target_dist in target_distances:
            # 해당 거리에 있는 세그먼트 찾기
            idx = np.searchsorted(cumulative_lengths, target_dist, side='right') - 1
            idx = max(0, min(idx, len(points) - 2))

            # 세그먼트 내 위치 계산
            seg_start = cumulative_lengths[idx]
            seg_length = segment_lengths[idx] if idx < len(segment_lengths) else 0

            if seg_length > 0:
                t = (target_dist - seg_start) / seg_length
                t = max(0, min(1, t))
                p = points[idx] * (1 - t) + points[idx + 1] * t
            else:
                p = points[idx]

            resampled.append([int(round(p[0])), int(round(p[1]))])

        # 중복 제거
        return self._remove_duplicate_points(resampled)
