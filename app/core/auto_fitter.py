"""
차선 라벨링 데이터에 맞춰 카메라 파라미터를 자동으로 피팅하는 모듈
"""
import copy
import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def densify_polyline(points, density=2.0):
    """폴리라인의 점들을 지정된 밀도(픽셀 단위)로 촘촘하게 채움"""
    if len(points) < 2:
        return points
    
    points = np.array(points)
    densified = []
    
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
             densified.append(p1)
             continue
             
        num_intervals = int(np.ceil(dist / density))
        if num_intervals < 1:
            num_intervals = 1
            
        # 점의 개수 = 구간 개수 + 1
        num_points = num_intervals + 1
        
        # 마지막 세그먼트인 경우 끝점 포함
        if i == len(points) - 2:
            segment_pts = np.linspace(p1, p2, num_points)
        else:
            segment_pts = np.linspace(p1, p2, num_points, endpoint=False)
            
        densified.extend(segment_pts)
        
    return densified


def _create_cost_function(projector, camera_params, lane_pts, iteration_count):
    """비용 함수 생성 (평균 거리 반환)"""
    opt_params = copy.deepcopy(camera_params)

    def cost_function(params):
        yaw, pitch, roll, f = params
        iteration_count[0] += 1

        opt_params.yaw = yaw
        opt_params.pitch = pitch
        opt_params.roll = roll
        opt_params.fx = f
        opt_params.fy = f

        try:
            # 실수형 좌표 투영 사용 (최적화용)
            projected_layers = projector.projection(opt_params, target_layers=['b2'], as_float=True, include_cache=False)

            if not projected_layers or 'b2' not in projected_layers:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e10 (투영 실패: 레이어 없음), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return 1e10

            b2_lines = projected_layers['b2']
            if not b2_lines:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e10 (투영 실패: b2 라인 없음), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return 1e10

            projected_pts = []
            for line in b2_lines:
                # 밀도 보강 (정밀도 향상)
                dense_line = densify_polyline(line, density=2.0)
                for pt in dense_line:
                    projected_pts.append(pt)

            if not projected_pts:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e10 (투영 실패: 점 없음), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return 1e10

            projected_pts = np.array(projected_pts, dtype=np.float64)

            # 화면 밖의 점 필터링 (마진 없음: 0px)
            # 해상도 기본값 1920x1080 사용 (사용자 요청)
            w = getattr(opt_params, 'resolution_width', 0) or 1920
            h = getattr(opt_params, 'resolution_height', 0) or 1080
            margin = 0

            mask = (
                (projected_pts[:, 0] >= -margin) & 
                (projected_pts[:, 0] <= w + margin) & 
                (projected_pts[:, 1] >= -margin) & 
                (projected_pts[:, 1] <= h + margin)
            )
            filtered_pts = projected_pts[mask]

            if len(filtered_pts) == 0:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e10 (필터링됨: 모든 점이 화면 밖), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return 1e10

            # KD-Tree를 사용한 최적화된 거리 계산 (O(n log m))
            tree = cKDTree(filtered_pts)
            min_distances, _ = tree.query(lane_pts, k=1)
            cost = np.mean(min_distances)

            if iteration_count[0] % 20 == 1:
                print(f"[auto_fitter] iter {iteration_count[0]}: cost={cost:.2f}, params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f}), n_pts={len(filtered_pts)}(raw:{len(projected_pts)})")

            return cost

        except Exception as e:
            print(f"[auto_fitter] 예외: {e}")
            return 1e10

    return cost_function


def _create_residual_function(projector, camera_params, lane_pts, iteration_count):
    """잔차 함수 생성 (LM용 - 각 점의 거리 벡터 반환)"""
    opt_params = copy.deepcopy(camera_params)

    def residual_function(params):
        yaw, pitch, roll, f = params
        iteration_count[0] += 1

        opt_params.yaw = yaw
        opt_params.pitch = pitch
        opt_params.roll = roll
        opt_params.fx = f
        opt_params.fy = f

        try:
            # 실수형 좌표 투영 사용 (최적화용)
            projected_layers = projector.projection(opt_params, target_layers=['b2'], as_float=True, include_cache=False)

            if not projected_layers or 'b2' not in projected_layers:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e5 (투영 실패: 레이어 없음), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return np.full(len(lane_pts), 1e5)

            b2_lines = projected_layers['b2']
            if not b2_lines:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e5 (투영 실패: b2 라인 없음), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return np.full(len(lane_pts), 1e5)

            projected_pts = []
            for line in b2_lines:
                # 밀도 보강 (정밀도 향상)
                dense_line = densify_polyline(line, density=2.0)
                for pt in dense_line:
                    projected_pts.append(pt)

            if not projected_pts:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e5 (투영 실패: 점 없음), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return np.full(len(lane_pts), 1e5)

            projected_pts = np.array(projected_pts, dtype=np.float64)

            # 화면 밖의 점 필터링 (마진 없음: 0px)
            # 해상도 기본값 1920x1080 사용 (사용자 요청)
            w = getattr(opt_params, 'resolution_width', 0) or 1920
            h = getattr(opt_params, 'resolution_height', 0) or 1080
            margin = 0

            mask = (
                (projected_pts[:, 0] >= -margin) & 
                (projected_pts[:, 0] <= w + margin) & 
                (projected_pts[:, 1] >= -margin) & 
                (projected_pts[:, 1] <= h + margin)
            )
            filtered_pts = projected_pts[mask]

            if len(filtered_pts) == 0:
                if iteration_count[0] % 20 == 1:
                     print(f"[auto_fitter] iter {iteration_count[0]}: cost=1e5 (필터링됨: 모든 점이 화면 밖), params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f})")
                return np.full(len(lane_pts), 1e5)

            # KD-Tree를 사용한 최적화된 거리 계산 (O(n log m))
            tree = cKDTree(filtered_pts)
            min_distances, _ = tree.query(lane_pts, k=1)

            if iteration_count[0] % 20 == 1:
                print(f"[auto_fitter] iter {iteration_count[0]}: mean_dist={np.mean(min_distances):.2f}, params=(y:{yaw:.2f}, p:{pitch:.2f}, r:{roll:.2f}, f:{f:.1f}), n_pts={len(filtered_pts)}(raw:{len(projected_pts)})")

            return min_distances

        except Exception as e:
            print(f"[auto_fitter] 예외: {e}")
            return np.full(len(lane_pts), 1e5)

    return residual_function


def _prepare_optimization(projector, camera_params, lane_points):
    """최적화 준비 (공통)"""
    if not lane_points or len(lane_points) < 2:
        print("[auto_fitter] 라벨링 점이 부족합니다")
        return None

    lane_pts = np.array(lane_points, dtype=np.float64)
    print(f"[auto_fitter] 라벨링 점 개수: {len(lane_pts)}")

    initial_params = np.array([
        camera_params.yaw,
        camera_params.pitch,
        camera_params.roll,
        camera_params.fx  # fx를 기준으로 f 통합 (fx = fy)
    ])
    print(f"[auto_fitter] 초기값: yaw={initial_params[0]:.2f}, pitch={initial_params[1]:.2f}, "
          f"roll={initial_params[2]:.2f}, f={initial_params[3]:.1f}")

    bounds = [
        (initial_params[0] - 20, initial_params[0] + 20),  # yaw
        (initial_params[1] - 20, initial_params[1] + 20),  # pitch
        (initial_params[2] - 1, initial_params[2] + 1),    # roll (PTZ 카메라: 거의 고정)
        (max(100, initial_params[3] * 0.8), initial_params[3] * 1.2),  # f (fx = fy)
    ]
    
    print("[auto_fitter] 파라미터 검색 범위 (Bounds):")
    print(f"  - Yaw:   {bounds[0][0]:.2f} ~ {bounds[0][1]:.2f}")
    print(f"  - Pitch: {bounds[1][0]:.2f} ~ {bounds[1][1]:.2f}")
    print(f"  - Roll:  {bounds[2][0]:.2f} ~ {bounds[2][1]:.2f}")
    print(f"  - F:     {bounds[3][0]:.1f} ~ {bounds[3][1]:.1f}")

    return lane_pts, initial_params, bounds


def _extract_result(result, method_name):
    """최적화 결과 추출"""
    optimized = result.x
    f = optimized[3]
    print(f"[auto_fitter] {method_name} 완료: yaw={optimized[0]:.2f}, pitch={optimized[1]:.2f}, "
          f"roll={optimized[2]:.2f}, f={f:.1f}")
    
    # 변화량 출력 (참고용)
    # Note: initial_params는 이 함수 스코프에 없으므로, fit 함수의 반환값이나 로그를 통해 확인해야 함.
    # 하지만 사용성 편의를 위해 로그를 남겨두는 패턴. (상위 로직에서 초기값을 이미 찍음)

    return {
        'yaw': optimized[0],
        'pitch': optimized[1],
        'roll': optimized[2],
        'fx': f,
        'fy': f
    }


def fit_powell(projector, camera_params, lane_points):
    """Powell 알고리즘으로 최적화 (방향 탐색 기반)"""
    prep = _prepare_optimization(projector, camera_params, lane_points)
    if prep is None:
        return None

    lane_pts, initial_params, bounds = prep
    iteration_count = [0]
    cost_fn = _create_cost_function(projector, camera_params, lane_pts, iteration_count)

    # 초기 비용 확인
    dummy_count = [2]
    temp_cost_fn = _create_cost_function(projector, camera_params, lane_pts, dummy_count)
    initial_cost = temp_cost_fn(initial_params)
    print(f"[auto_fitter] 초기 상태(수정 전) Cost: {initial_cost:.2f}")

    print("[auto_fitter] Powell 알고리즘 시작...")
    result = minimize(
        cost_fn,
        initial_params,
        method='Powell',
        bounds=bounds,
        options={'maxiter': 500, 'ftol': 0.05}
    )

    print(f"[auto_fitter] iterations={result.nit}, final_cost={result.fun:.2f}")
    return _extract_result(result, "Powell")


def fit_nelder_mead(projector, camera_params, lane_points):
    """Nelder-Mead 알고리즘으로 최적화 (심플렉스 기반, 노이즈에 강함)"""
    prep = _prepare_optimization(projector, camera_params, lane_points)
    if prep is None:
        return None

    lane_pts, initial_params, bounds = prep
    iteration_count = [0]
    cost_fn = _create_cost_function(projector, camera_params, lane_pts, iteration_count)

    # 초기 비용 확인
    dummy_count = [2]
    temp_cost_fn = _create_cost_function(projector, camera_params, lane_pts, dummy_count)
    initial_cost = temp_cost_fn(initial_params)
    print(f"[auto_fitter] 초기 상태(수정 전) Cost: {initial_cost:.2f}")

    print("[auto_fitter] NM 알고리즘 시작...")
    result = minimize(
        cost_fn,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.05}
    )

    # 경계 적용 (Nelder-Mead는 경계를 직접 지원하지 않음)
    optimized = result.x
    for i, (low, high) in enumerate(bounds):
        optimized[i] = np.clip(optimized[i], low, high)
    result.x = optimized

    print(f"[auto_fitter] iterations={result.nit}, final_cost={result.fun:.2f}")
    return _extract_result(result, "NM")


def fit_lm(projector, camera_params, lane_points):
    """Levenberg-Marquardt 알고리즘으로 최적화 (비선형 최소제곱, 로봇 비전 표준)"""
    prep = _prepare_optimization(projector, camera_params, lane_points)
    if prep is None:
        return None

    lane_pts, initial_params, bounds = prep
    iteration_count = [0]
    residual_fn = _create_residual_function(projector, camera_params, lane_pts, iteration_count)

    # 초기 비용 확인
    dummy_count = [2]
    temp_residual_fn = _create_residual_function(projector, camera_params, lane_pts, dummy_count)
    initial_residuals = temp_residual_fn(initial_params)
    initial_cost = np.mean(initial_residuals)
    print(f"[auto_fitter] 초기 상태(수정 전) Cost: {initial_cost:.2f}")

    # least_squares용 경계 형식 변환
    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    print("[auto_fitter] Levenberg-Marquardt 알고리즘 시작...")
    result = least_squares(
        residual_fn,
        initial_params,
        method='trf',  # Trust Region Reflective (LM 변형, 경계 지원)
        bounds=(lower_bounds, upper_bounds),
        ftol=1e-10,       # 비용 함수 변화 허용 오차 (매우 정밀하게)
        xtol=1e-10,       # 파라미터 변화 허용 오차
        gtol=1e-10,       # 그레디언트 노름 허용 오차
        x_scale='jac',    # 파라미터 스케일 자동 조정 (각도 vs 픽셀 단위 차이 보정)
        loss='soft_l1',   # 이상치(Outlier)에 강건한 손실 함수 사용
        max_nfev=500,
        verbose=1
    )

    final_cost = np.mean(result.fun)
    print(f"[auto_fitter] nfev={result.nfev}, final_mean_dist={final_cost:.2f}")
    return _extract_result(result, "LM")


def fit_differential_evolution(projector, camera_params, lane_points):
    """Differential Evolution 알고리즘으로 최적화 (진화 알고리즘 기반, 글로벌 최적화)"""
    prep = _prepare_optimization(projector, camera_params, lane_points)
    if prep is None:
        return None

    lane_pts, initial_params, bounds = prep
    iteration_count = [0]
    cost_fn = _create_cost_function(projector, camera_params, lane_pts, iteration_count)

    # 초기 비용 확인
    dummy_count = [2]
    temp_cost_fn = _create_cost_function(projector, camera_params, lane_pts, dummy_count)
    initial_cost = temp_cost_fn(initial_params)
    print(f"[auto_fitter] 초기 상태(수정 전) Cost: {initial_cost:.2f}")

    print("[auto_fitter] Differential Evolution 알고리즘 시작...")
    result = differential_evolution(
        cost_fn,
        bounds=bounds,
        seed=42,             # 재현성 보장
        maxiter=500,         # 최대 세대 수
        popsize=15,          # 개체군 크기 (파라미터 수 × 3-4)
        atol=0.05,           # 절대 허용 오차
        tol=0.05,            # 상대 허용 오차
        workers=1,           # 단일 스레드 (안정성 우선, 병렬 버전은 fit_differential_evolution_parallel 사용)
        updating='deferred', # 비동기 업데이트 (수렴 속도 향상)
        strategy='best1bin', # 기본 전략 (균형잡힌 성능)
        mutation=(0.5, 1.5), # 변이 상수 범위
        recombination=0.7    # 재조합 확률
    )

    print(f"[auto_fitter] iterations={result.nit}, final_cost={result.fun:.2f}")
    return _extract_result(result, "Differential Evolution")


# ============================================================
# 병렬 처리 버전 (Pickle 가능한 구현)
# ============================================================

# 전역 변수 (멀티프로세싱을 위해 필요)
_global_projector = None
_global_camera_params = None
_global_lane_pts = None


def _parallel_cost_function(params):
    """병렬 처리 가능한 전역 cost function (pickle 가능)"""
    global _global_projector, _global_camera_params, _global_lane_pts
    
    yaw, pitch, roll, f = params
    opt_params = copy.deepcopy(_global_camera_params)
    
    opt_params.yaw = yaw
    opt_params.pitch = pitch
    opt_params.roll = roll
    opt_params.fx = f
    opt_params.fy = f

    try:
        # 실수형 좌표 투영 사용 (최적화용)
        projected_layers = _global_projector.projection(opt_params, target_layers=['b2'], as_float=True, include_cache=False)

        if not projected_layers or 'b2' not in projected_layers:
            return 1e10

        b2_lines = projected_layers['b2']
        if not b2_lines:
            return 1e10

        projected_pts = []
        for line in b2_lines:
            # 밀도 보강 (정밀도 향상)
            dense_line = densify_polyline(line, density=2.0)
            for pt in dense_line:
                projected_pts.append(pt)

        if not projected_pts:
            return 1e10

        projected_pts = np.array(projected_pts, dtype=np.float64)

        # 화면 밖의 점 필터링 (마진 없음: 0px)
        w = getattr(opt_params, 'resolution_width', 0) or 1920
        h = getattr(opt_params, 'resolution_height', 0) or 1080
        margin = 0

        mask = (
            (projected_pts[:, 0] >= -margin) & 
            (projected_pts[:, 0] <= w + margin) & 
            (projected_pts[:, 1] >= -margin) & 
            (projected_pts[:, 1] <= h + margin)
        )
        filtered_pts = projected_pts[mask]

        if len(filtered_pts) == 0:
            return 1e10

        # KD-Tree를 사용한 최적화된 거리 계산
        tree = cKDTree(filtered_pts)
        min_distances, _ = tree.query(_global_lane_pts, k=1)
        cost = np.mean(min_distances)

        return cost

    except Exception as e:
        print(f"[auto_fitter] 병렬 처리 예외: {e}")
        return 1e10


def fit_differential_evolution_parallel(projector, camera_params, lane_points, workers=16):
    """Differential Evolution 알고리즘으로 최적화 (병렬 처리 버전)
    
    Args:
        projector: MapProjector 객체
        camera_params: CameraParams 객체
        lane_points: 라벨링된 차선 점 리스트
        workers: 사용할 CPU 코어 수 (기본값: 16, -1이면 모든 코어 사용)
    """
    global _global_projector, _global_camera_params, _global_lane_pts
    
    prep = _prepare_optimization(projector, camera_params, lane_points)
    if prep is None:
        return None

    lane_pts, initial_params, bounds = prep
    
    # 전역 변수에 할당 (멀티프로세싱을 위해)
    _global_projector = projector
    _global_camera_params = camera_params
    _global_lane_pts = lane_pts

    # 초기 비용 확인
    initial_cost = _parallel_cost_function(initial_params)
    print(f"[auto_fitter] 초기 상태(수정 전) Cost: {initial_cost:.2f}")

    print(f"[auto_fitter] Differential Evolution (병렬) 알고리즘 시작... (workers={workers})")
    result = differential_evolution(
        _parallel_cost_function,
        bounds=bounds,
        seed=42,             # 재현성 보장
        maxiter=100,         # 최대 세대 수 (실제로는 조기 종료로 더 적게 실행됨)
        popsize=10,          # 개체군 크기 (빠른 수렴을 위해 감소)
        atol=0.05,           # 절대 허용 오차
        tol=0.05,            # 상대 허용 오차
        workers=workers,     # 병렬 처리 활성화!
        updating='deferred', # 비동기 업데이트 (수렴 속도 향상)
        strategy='best1bin', # 기본 전략 (균형잡힌 성능)
        mutation=(0.5, 1.5), # 변이 상수 범위
        recombination=0.7    # 재조합 확률
    )

    print(f"[auto_fitter] iterations={result.nit}, final_cost={result.fun:.2f}")
    
    # 전역 변수 정리
    _global_projector = None
    _global_camera_params = None
    _global_lane_pts = None
    
    return _extract_result(result, "Differential Evolution (병렬)")
