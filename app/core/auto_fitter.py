"""
차선 라벨링 데이터에 맞춰 카메라 파라미터를 자동으로 피팅하는 모듈
"""
import copy
import numpy as np
from scipy.optimize import minimize, least_squares, differential_evolution
from scipy.spatial.distance import cdist


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
            projected_layers = projector.projection_float(opt_params, target_layers=['b2'])

            if not projected_layers or 'b2' not in projected_layers:
                return 1e10

            b2_lines = projected_layers['b2']
            if not b2_lines:
                return 1e10

            projected_pts = []
            for line in b2_lines:
                for pt in line:
                    projected_pts.append(pt)

            if not projected_pts:
                return 1e10

            projected_pts = np.array(projected_pts, dtype=np.float64)
            distances = cdist(lane_pts, projected_pts, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            cost = np.mean(min_distances)

            if iteration_count[0] % 20 == 1:
                print(f"[auto_fitter] iter {iteration_count[0]}: cost={cost:.2f}")

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
            projected_layers = projector.projection_float(opt_params, target_layers=['b2'])

            if not projected_layers or 'b2' not in projected_layers:
                return np.full(len(lane_pts), 1e5)

            b2_lines = projected_layers['b2']
            if not b2_lines:
                return np.full(len(lane_pts), 1e5)

            projected_pts = []
            for line in b2_lines:
                for pt in line:
                    projected_pts.append(pt)

            if not projected_pts:
                return np.full(len(lane_pts), 1e5)

            projected_pts = np.array(projected_pts, dtype=np.float64)
            distances = cdist(lane_pts, projected_pts, metric='euclidean')
            min_distances = np.min(distances, axis=1)

            if iteration_count[0] % 20 == 1:
                print(f"[auto_fitter] iter {iteration_count[0]}: mean_dist={np.mean(min_distances):.2f}")

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

    return lane_pts, initial_params, bounds


def _extract_result(result, method_name):
    """최적화 결과 추출"""
    optimized = result.x
    f = optimized[3]
    print(f"[auto_fitter] {method_name} 완료: yaw={optimized[0]:.2f}, pitch={optimized[1]:.2f}, "
          f"roll={optimized[2]:.2f}, f={f:.1f}")

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

    print("[auto_fitter] Nelder-Mead 알고리즘 시작...")
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
    return _extract_result(result, "Nelder-Mead")


def fit_lm(projector, camera_params, lane_points):
    """Levenberg-Marquardt 알고리즘으로 최적화 (비선형 최소제곱, 로봇 비전 표준)"""
    prep = _prepare_optimization(projector, camera_params, lane_points)
    if prep is None:
        return None

    lane_pts, initial_params, bounds = prep
    iteration_count = [0]
    residual_fn = _create_residual_function(projector, camera_params, lane_pts, iteration_count)

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

    print("[auto_fitter] Differential Evolution 알고리즘 시작...")
    result = differential_evolution(
        cost_fn,
        bounds=bounds,
        seed=42,             # 재현성 보장
        maxiter=500,         # 최대 세대 수
        popsize=15,          # 개체군 크기 (파라미터 수 × 3-4)
        atol=0.05,           # 절대 허용 오차
        tol=0.05,            # 상대 허용 오차
        workers=1,           # 단일 스레드 (안정성)
        updating='deferred', # 비동기 업데이트 (수렴 속도 향상)
        strategy='best1bin', # 기본 전략 (균형잡힌 성능)
        mutation=(0.5, 1.5), # 변이 상수 범위
        recombination=0.7    # 재조합 확률
    )

    print(f"[auto_fitter] iterations={result.nit}, final_cost={result.fun:.2f}")
    return _extract_result(result, "Differential Evolution")
