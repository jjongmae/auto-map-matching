"""
도로면 모델 생성 및 관련 처리를 위한 모듈.
"""
import numpy as np
import geopandas as gpd
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay

def create_road_surface_model(gdfs: list[gpd.GeoDataFrame], tolerance: float = 0.01) -> tuple:
    """
    단순화된 지오메트리의 모든 정점을 사용하여 도로면 보간 모델과
    해석적 교차 계산을 위한 3D 삼각망을 생성합니다.

    Args:
        gdfs: 3D 라인스트링 지오메트리를 포함하는 GeoDataFrame의 리스트.
        tolerance: 지오메트리 단순화에 사용할 허용 오차.

    Returns:
        (interpolator, triangulation, vertices) 튜플.
        생성 실패 시 (None, None, None)을 반환합니다.
    """
    all_points = []
    for gdf in gdfs:
        if gdf.empty: continue
        simplified_geometries = gdf.geometry.simplify(tolerance, preserve_topology=True)
        for geom in simplified_geometries:
            if geom.is_empty: continue
            geoms_to_process = geom.geoms if hasattr(geom, 'geoms') else [geom]
            for line in geoms_to_process:
                coords = np.asarray(line.coords)
                if coords.shape[1] == 3:
                    all_points.append(coords)

    if not all_points:
        print("[WARNING] Surface: 도로면 모델을 생성하기 위한 3D 정점이 부족합니다.")
        return None, None, None

    try:
        vertices = np.vstack(all_points)
        _, indices = np.unique(vertices[:, :2], axis=0, return_index=True)
        unique_vertices = vertices[indices]
        points = unique_vertices[:, :2]
        values = unique_vertices[:, 2]
        
        interpolator = LinearNDInterpolator(points, values)
        triangulation = Delaunay(points)
        
        print(f"[INFO] Surface: 도로면 모델 생성 성공 - {len(unique_vertices)}개의 고유 정점, {len(triangulation.simplices)}개의 삼각형")
        return interpolator, triangulation, unique_vertices
    except Exception as e:
        print(f"[ERROR] Surface: 도로면 모델 생성 중 오류 발생 - {e}")
        return None, None, None