"""
Shapefile 데이터 로딩을 담당하는 모듈.
"""
from pathlib import Path
import geopandas as gpd
from shapely.geometry import Point

def load_shp_layers(layer_config: dict[str, str], center_point: Point, radius: int) -> dict[str, gpd.GeoDataFrame]:
    """
    설정에 따라 여러 Shapefile을 로드하고, 중심점 반경 기준으로 필터링합니다.

    Args:
        layer_config: 로드할 레이어의 이름과 파일 경로를 담은 딕셔너리.
                      예: {'roads': 'data/A2.shp', 'lanes': 'data/B2.shp'}
        center_point: 검색 중심점 (Shapely Point 객체).
        radius: 검색 반경 (미터 단위).

    Returns:
        레이어 이름을 키로, 필터링된 GeoDataFrame을 값으로 갖는 딕셔너리.
    """
    search_area = center_point.buffer(radius)
    loaded_data = {}

    for name, shp_path_str in layer_config.items():
        shp_path = Path(shp_path_str)
        if not shp_path.exists():
            print(f"[경고] Loader: 레이어 '{name}' - {shp_path}를 찾을 수 없습니다.")
            loaded_data[name] = gpd.GeoDataFrame()
            continue
        try:
            gdf = gpd.read_file(shp_path, encoding='cp949')
            sindex = gdf.sindex
            possible_matches_index = list(sindex.intersection(search_area.bounds))
            possible_matches = gdf.iloc[possible_matches_index]
            intersecting = possible_matches[possible_matches.intersects(search_area)].copy()
            print(f"[정보] Loader: 레이어 '{name}' - {radius}m 반경 내에서 {len(intersecting)}개의 객체를 로드했습니다.")
            loaded_data[name] = intersecting
        except Exception as e:
            print(f"[오류] Loader: 레이어 '{name}' ({shp_path}) 처리 중 오류 - {e}")
            loaded_data[name] = gpd.GeoDataFrame()

    return loaded_data
