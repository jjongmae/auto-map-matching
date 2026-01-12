"""
지도 데이터를 카메라 뷰로 투영하고, 픽셀 좌표를 월드 좌표로 역투영하는 메인 클래스.
"""
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from pathlib import Path
import dataclasses
from typing import List, Dict

from . import loader
from . import geometry
from . import surface
from .camera import CameraParams

@dataclasses.dataclass
class ProjectionResult:
    """
    투영 결과를 담는 데이터 클래스.

    Attributes:
        projected_layers: 2D로 투영된 레이어 데이터.
        unprojection_cache: 역투영 시 사용될 캐시 데이터.
    """
    projected_layers: Dict[str, List[np.ndarray]]
    unprojection_cache: List[Dict]

class MapProjector:
    def __init__(self, shp_root: str | Path = "shp"):
        """
        MapProjector 클래스의 생성자입니다.

        Args:
            shp_root: Shapefile이 저장된 루트 디렉토리 경로.
        """
        self.shp_root = Path(shp_root)
        self.layers = {}
        self.road_surface_interpolator = None
        self.road_surface_triangulation = None
        self.road_surface_vertices = None
        self.a2_segments_cache = None  # A2 세그먼트 캐시

    def load_map_data(self, x: float, y: float, radius: int = 500, simplification_tolerance: float = 0.01, layer_config: dict | None = None):
        """
        주어진 좌표 반경 내의 Shapefile을 로드하고 3D 도로면 모델을 생성합니다.

        Args:
            x: 중심 좌표의 X값.
            y: 중심 좌표의 Y값.
            radius: 검색 반경 (미터).
            simplification_tolerance: 도로면 생성 시 지오메트리 단순화 허용 오차.
            layer_config: 로드할 레이어 설정. None이면 기본값을 사용합니다.
        """
        center_point = Point(x, y)
        
        if layer_config is None:
            layer_config = {
                "a2": str(self.shp_root / "A2_LINK.shp"),
                "b2": str(self.shp_root / "B2_SURFACELINEMARK.shp"),
            }

        self.layers = loader.load_shp_layers(layer_config, center_point, radius)

        # A2 세그먼트 사전 처리
        self._preprocess_a2_segments()

        surface_gdfs = [self.layers.get('a2'), self.layers.get('b2')]
        self.road_surface_interpolator, self.road_surface_triangulation, self.road_surface_vertices = \
            surface.create_road_surface_model([gdf for gdf in surface_gdfs if gdf is not None], simplification_tolerance)

    def projection(self, params: CameraParams) -> ProjectionResult:
        """
        로드된 지도 레이어와 도로면을 2D 픽셀로 투영하고, 역투영 캐시를 포함한 결과를 반환합니다.
        """
        C = np.array([params.x, params.y, params.z], np.float32)
        R_cam = geometry.cam_rotation(params.yaw, params.pitch, params.roll)
        R_tot = R_cam @ geometry.B_ENU2CV
        
        projected_layers = {"b2": [], "road_surface": []}
        unprojection_cache = []
        
        # Define screen boundaries with a margin
        margin = 2000
        # CameraParams에서 해상도 값을 직접 가져오도록 수정
        min_u, max_u = -margin, params.resolution_width + margin
        min_v, max_v = -margin, params.resolution_height + margin
        
        layers_to_project = {"b2": self.layers.get("b2")}
        for name, gdf in layers_to_project.items():
            if gdf is None or gdf.empty: continue
            for geom in gdf.geometry:
                geoms = geom.geoms if hasattr(geom, 'geoms') else [geom.exterior if isinstance(geom, Polygon) else geom]
                for line in geoms:
                    if line is None or line.is_empty: continue
                    pts_w = np.asarray(line.coords, np.float32)
                    if pts_w.shape[0] < 2: continue
                    if pts_w.shape[1] == 2 and self.road_surface_interpolator:
                        z = self.road_surface_interpolator(pts_w)
                        pts_w = np.hstack([pts_w, np.nan_to_num(z)[:, np.newaxis]])
                    
                    Pc = (pts_w - C) @ R_tot.T
                    z_cam = Pc[:, 2]
                    mask = (z_cam > params.near) & (z_cam < params.far)
                    if not np.any(mask): continue

                    x_n = Pc[mask, 0] / z_cam[mask]
                    y_n = Pc[mask, 1] / z_cam[mask]
                    x_d, y_d = geometry.distort(x_n, y_n, params.k1, params.k2, params.p1, params.p2, params.k3)
                    u = params.fx * x_d + params.cx
                    v = params.fy * y_d + params.cy
                    
                    # --- 수정된 필터링 로직 ---
                    # 1. 먼저 부동소수점 좌표 배열 생성
                    points_2d_float = np.stack([u, v], axis=1)
                    
                    # 2. NaN/Inf 필터링
                    finite_mask = np.all(np.isfinite(points_2d_float), axis=1)
                    finite_points = points_2d_float[finite_mask]
                    
                    if finite_points.shape[0] == 0: continue

                    # 3. 화면 경계 필터링 (유효한 부동소수점 값으로 수행)
                    boundary_mask = (
                        (finite_points[:, 0] >= min_u) & (finite_points[:, 0] <= max_u) &
                        (finite_points[:, 1] >= min_v) & (finite_points[:, 1] <= max_v)
                    )
                    final_points = finite_points[boundary_mask]

                    # 4. 모든 필터링을 통과한 최종 데이터만 정수로 변환
                    if final_points.shape[0] > 0:
                        projected_layers[name].append(final_points.astype(np.int32))

        if self.road_surface_triangulation and self.road_surface_vertices is not None:
            tri_vertices = self.road_surface_vertices[self.road_surface_triangulation.simplices]
            for i, tri in enumerate(tri_vertices):
                Pc = (tri - C) @ R_tot.T
                z_cam = Pc[:, 2]
                if not np.all((z_cam > params.near) & (z_cam < params.far)): continue

                x_n = Pc[:, 0] / z_cam
                y_n = Pc[:, 1] / z_cam
                x_d, y_d = geometry.distort(x_n, y_n, params.k1, params.k2, params.p1, params.p2, params.k3)
                u = params.fx * x_d + params.cx
                v = params.fy * y_d + params.cy
                
                p_2d_raw_float = np.stack([u, v], axis=1)

                # --- 수정된 필터링 로직 (road_surface) ---
                # 1. NaN/Inf 필터링
                finite_mask = np.all(np.isfinite(p_2d_raw_float), axis=1)
                
                # 2. 온전한 삼각형(점 3개)이 모두 유효한지 먼저 확인
                if np.count_nonzero(finite_mask) == 3:
                    p_2d_float = p_2d_raw_float # 이 경우 모든 점이 유효함
                    
                    # 3. 화면 경계 필터링
                    boundary_mask = (
                        (p_2d_float[:, 0] >= min_u) & (p_2d_float[:, 0] <= max_u) &
                        (p_2d_float[:, 1] >= min_v) & (p_2d_float[:, 1] <= max_v)
                    )
                    
                    # 4. 경계 필터링 후에도 온전한 삼각형인지 재확인
                    if np.count_nonzero(boundary_mask) == 3:
                        final_triangle = p_2d_float.astype(np.int32)
                        projected_layers["road_surface"].append(final_triangle)
                        for j in range(3):
                            unprojection_cache.append({
                                '2d_line_segment': (final_triangle[j], final_triangle[(j + 1) % 3]),
                                '3d_line_segment': (tri[j], tri[(j + 1) % 3]),
                                'source_triangle_index': i
                            })
        return ProjectionResult(projected_layers=projected_layers, unprojection_cache=unprojection_cache)

    def unprojection(self, u: float, v: float, params: CameraParams, proj_result: ProjectionResult) -> np.ndarray | None:
        """
        2D 픽셀을 3D 월드 좌표로 역투영합니다.

        Args:
            u: 픽셀 u 좌표.
            v: 픽셀 v 좌표.
            params: 카메라 파라미터.
            proj_result: projection 메소드로부터 반환된 ProjectionResult 객체.
        """
        if not self.road_surface_triangulation or self.road_surface_vertices is None:
            print(f"[오류] MapProjector: 도로면 모델이 없습니다. Unprojection을 수행할 수 없습니다.")
            return None

        C = np.array([params.x, params.y, params.z], np.float32)
        R_cam = geometry.cam_rotation(params.yaw, params.pitch, params.roll)
        R_inv = np.linalg.inv(R_cam @ geometry.B_ENU2CV)

        x_d, y_d = (u - params.cx) / params.fx, (v - params.cy) / params.fy
        x_n, y_n = geometry.undistort(x_d, y_d, params.k1, params.k2, params.p1, params.p2, params.k3)
        
        ray_dir_cam = np.array([x_n, y_n, 1.0], dtype=np.float32)
        ray_dir_world = ray_dir_cam @ R_inv.T
        ray_dir_world /= np.linalg.norm(ray_dir_world)

        triangles = self.road_surface_vertices[self.road_surface_triangulation.simplices]
        intersect_point = geometry.ray_mesh_intersect(C, ray_dir_world, triangles)

        if intersect_point is not None:
            return intersect_point

        if not proj_result.unprojection_cache:
            print(f"[경고] MapProjector: Unprojection 추론 실패 - 투영 캐시가 비어있습니다.")
            return None

        click_p = np.array([u, v])

        # 벡터화된 거리 계산
        p1_array = np.array([item['2d_line_segment'][0] for item in proj_result.unprojection_cache])  # (N, 2)
        p2_array = np.array([item['2d_line_segment'][1] for item in proj_result.unprojection_cache])  # (N, 2)

        # 모든 선분과의 거리를 한번에 계산
        distances = geometry.point_to_segments_dist_vectorized(click_p, p1_array, p2_array)

        # 최소 거리 인덱스 찾기
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]
        closest_cache_item = proj_result.unprojection_cache[min_idx]

        
        if closest_cache_item is None:
            print(f"[경고] MapProjector: Unprojection 추론 실패 - 가장 가까운 도로 경계선을 찾을 수 없습니다.")
            return None

        p2d_1, p2d_2 = map(np.array, closest_cache_item['2d_line_segment'])
        v3d_1, v3d_2 = map(np.array, closest_cache_item['3d_line_segment'])

        l2 = np.sum((p2d_1 - p2d_2)**2)
        t_param = 0.0 if l2 == 0.0 else max(0, min(1, np.dot(click_p - p2d_1, p2d_2 - p2d_1) / l2))
        
        road_edge_point_3d = v3d_1 + t_param * (v3d_2 - v3d_1)
        shoulder_plane_normal = np.array([0., 0., 1.])
        
        denom = np.dot(shoulder_plane_normal, ray_dir_world)
        if abs(denom) > 1e-6:
            s = np.dot(road_edge_point_3d - C, shoulder_plane_normal) / denom
            if s > 0:
                return C + s * ray_dir_world

        print(f"[경고] MapProjector: Unprojection 추론 실패 - 가상 갓길 평면과도 교차점을 찾을 수 없습니다.")
        return None

    def _preprocess_a2_segments(self):
        """
        A2 LineString을 개별 세그먼트(점-점 사이)로 분해하여 캐싱합니다.

        예: LineString([P1, P2, P3, P4])
        → 세그먼트: [P1-P2, P2-P3, P3-P4]

        캐시 구조:
        {
            'segment_ids': List[str],              # 각 세그먼트의 원본 ID (중복 가능)
            'start_points': np.ndarray (M, 2),     # 세그먼트 시작점들
            'end_points': np.ndarray (M, 2),       # 세그먼트 끝점들
            'directions': np.ndarray (M, 2),       # 정규화된 방향 벡터
            'lengths': np.ndarray (M,),            # 세그먼트 길이
            'segment_indices': np.ndarray (M,)     # 원본 LineString 내 인덱스
        }
        * M = 전체 세그먼트 개수 (LineString 개수보다 훨씬 많음)
        """
        if 'a2' not in self.layers or self.layers['a2'].empty:
            self.a2_segments_cache = None
            return

        a2_gdf = self.layers['a2']

        # 동적 리스트로 수집 (전체 세그먼트 개수를 미리 알 수 없음)
        segment_ids = []
        start_points_list = []
        end_points_list = []
        segment_indices_list = []
        lane_numbers_list = []  # 차선 번호 정보

        for row in a2_gdf.itertuples():
            geom = row.geometry
            link_id = str(row.ID)

            # 차선 번호 정보 가져오기 (LaneNo 필드)
            lane_no = getattr(row, 'LaneNo', None)

            # MultiLineString 처리
            if geom.geom_type == 'MultiLineString':
                lines = list(geom.geoms)
            else:
                lines = [geom]

            for line in lines:
                coords = np.array(line.coords, dtype=np.float32)

                # LineString의 연속된 점들을 세그먼트로 분해
                # [P1, P2, P3, P4] → [(P1,P2), (P2,P3), (P3,P4)]
                for i in range(len(coords) - 1):
                    start_points_list.append(coords[i][:2])
                    end_points_list.append(coords[i + 1][:2])
                    segment_ids.append(link_id)
                    segment_indices_list.append(i)
                    lane_numbers_list.append(lane_no)

        if not start_points_list:
            self.a2_segments_cache = None
            return

        # NumPy 배열로 변환
        start_points = np.array(start_points_list, dtype=np.float32)  # (M, 2)
        end_points = np.array(end_points_list, dtype=np.float32)      # (M, 2)

        # 방향 벡터 및 길이 계산 (벡터화)
        directions = end_points - start_points  # (M, 2)
        lengths = np.linalg.norm(directions, axis=1)  # (M,)

        # 정규화 (길이 0인 세그먼트 제외)
        valid_mask = lengths > 1e-6
        directions_normalized = np.zeros_like(directions)
        directions_normalized[valid_mask] = directions[valid_mask] / lengths[valid_mask, np.newaxis]

        self.a2_segments_cache = {
            'segment_ids': segment_ids,  # List[str]
            'start_points': start_points,
            'end_points': end_points,
            'directions': directions_normalized,
            'lengths': lengths,
            'segment_indices': np.array(segment_indices_list),
            'valid_mask': valid_mask,
            'lane_numbers': lane_numbers_list  # List[int or None] 차선 번호
        }

        print(f"[정보] MapProjector: A2 세그먼트 사전 처리 완료 - "
              f"{len(a2_gdf)}개 LineString → {len(start_points)}개 세그먼트")

    def find_nearest_a2_segment(self, x: float, y: float, max_distance: float = 5.0) -> dict | None:
        """
        벡터화 연산으로 가장 가까운 A2 세그먼트를 찾습니다.
        (LineString의 모든 작은 세그먼트 고려)

        Args:
            x: 월드 좌표 X (ENU)
            y: 월드 좌표 Y (ENU)
            max_distance: 최대 탐색 거리 (미터)

        Returns:
            {
                'segment_id': str,                    # A2 세그먼트 ID
                'distance': float,                    # 최단 거리 (미터)
                'nearest_point': np.ndarray (2,),     # 세그먼트 상의 가장 가까운 점 [x, y]
                'direction': np.ndarray (2,),         # 세그먼트 방향 벡터 (정규화)
                'segment_start': np.ndarray (2,),     # 세그먼트 시작점
                'segment_end': np.ndarray (2,)        # 세그먼트 끝점
            }
            또는 None (세그먼트를 찾지 못한 경우)
        """
        if self.a2_segments_cache is None:
            return None

        cache = self.a2_segments_cache
        query_point = np.array([x, y], dtype=np.float32)

        # 벡터화된 점-선분 거리 계산
        start_to_query = query_point - cache['start_points']  # (M, 2)
        start_to_end = cache['end_points'] - cache['start_points']  # (M, 2)

        dot_products = np.sum(start_to_query * start_to_end, axis=1)
        lengths_squared = np.sum(start_to_end * start_to_end, axis=1)

        t = np.zeros_like(dot_products)
        valid_lengths = lengths_squared > 1e-6
        t[valid_lengths] = np.clip(dot_products[valid_lengths] / lengths_squared[valid_lengths], 0.0, 1.0)

        nearest_points_on_segments = cache['start_points'] + t[:, np.newaxis] * start_to_end
        distances = np.linalg.norm(query_point - nearest_points_on_segments, axis=1)

        # 유효한 세그먼트 중 최소 거리
        valid_distances = distances.copy()
        valid_distances[~cache['valid_mask']] = np.inf

        if np.all(valid_distances == np.inf):
            return None

        min_idx = np.argmin(valid_distances)
        min_distance = valid_distances[min_idx]

        if min_distance > max_distance:
            return None

        return {
            'segment_id': cache['segment_ids'][min_idx],  # 원본 LineString ID
            'distance': float(min_distance),
            'nearest_point': nearest_points_on_segments[min_idx],
            'direction': cache['directions'][min_idx],     # 해당 세그먼트의 방향
            'segment_start': cache['start_points'][min_idx],
            'segment_end': cache['end_points'][min_idx],
            'segment_index': int(cache['segment_indices'][min_idx]),  # LineString 내 위치
            'lane_no': cache['lane_numbers'][min_idx]  # 차선 번호
        }