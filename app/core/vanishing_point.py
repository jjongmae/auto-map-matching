import cv2
import numpy as np
import math

RANSAC_RANDOM_SEED = 42

def detect_lines_lsd(image):
    """
    LSD (Line Segment Detector)를 사용하여 이미지에서 선분을 검출합니다.
    
    인자:
        image: 입력 이미지 (BGR 또는 Grayscale).
        
    반환:
        lines: 선분 [x1, y1, x2, y2]을 포함하는 형태 (N, 4)의 numpy 배열.
               선이 발견되지 않으면 None을 반환합니다.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # 기본 파라미터로 LSD 검출기 생성
    # LSD_REFINE_STD는 표준 정제입니다.
    lsd = cv2.createLineSegmentDetector(0) 
    
    lines, _, _, _ = lsd.detect(gray)
    
    if lines is not None:
        # LSD의 lines 포맷은 (N, 1, 4)이므로 (N, 4)로 변경
        lines = lines.reshape(-1, 4)
        # 매우 짧은 선분 제거 (선택 사항이지만 견고성에 좋음)
        min_len = 15  # 픽셀 단위 최소 길이
        keep = []
        for i, line in enumerate(lines):
            x1, y1, x2, y2 = line
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            if length > min_len:
                keep.append(i)
        
        if keep:
            lines = lines[keep]
        else:
            lines = None
            
    return lines

def compute_line_params(lines):
    """
    선분 [x1, y1, x2, y2]를 일반형 ax + by + c = 0으로 변환합니다.
    가중치 부여를 위한 길이와 중점도 반환합니다.
    """
    if lines is None or len(lines) == 0:
        return None
        
    # lines: (N, 4)
    x1 = lines[:, 0]
    y1 = lines[:, 1]
    x2 = lines[:, 2]
    y2 = lines[:, 3]
    
    # a = y1 - y2
    a = y1 - y2
    # b = x2 - x1
    b = x2 - x1
    # c = -ax1 - by1 = -(y1-y2)x1 - (x2-x1)y1 = -x1y1 + x2x1 - x2y1 + x1y1 = x2y1 - x1y2 ??
    # 일반형: a*x + b*y + c = 0
    # (x1, y1)과 (x2, y2)를 지나는 직선
    # (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)
    # (y - y1)(x2 - x1) = (x - x1)(y2 - y1)
    # y(x2 - x1) - y1(x2 - x1) = x(y2 - y1) - x1(y2 - y1)
    # (y1 - y2)x + (x2 - x1)y + x1(y2 - y1) - y1(x2 - x1) = 0
    # a = y1 - y2
    # b = x2 - x1
    # c = x1*y2 - x2*y1  (부호 확인)
    # -x1y1 + x1y2 + x1y1 - x2y1 = x1y2 - x2y1
    # 따라서 c = x1*y2 - x2*y1
    
    c = x1 * y2 - x2 * y1
    
    # 거리 계산을 위해 (a, b)를 단위 벡터로 정규화 distance = |ax+by+c| / sqrt(a^2+b^2)
    norm = np.sqrt(a**2 + b**2)
    
    # 점(길이 0인 선)에 대한 0으로 나누기 방지 - 필터링되었지만 안전장치
    norm = np.maximum(norm, 1e-6)
    
    a_norm = a / norm
    b_norm = b / norm
    c_norm = c / norm
    
    return np.column_stack((a_norm, b_norm, c_norm))

def get_intersection(line1, line2):
    """
    ax+by+c=0 형태의 (a, b, c)로 주어진 두 직선의 교차점을 찾습니다.
    (a1, b1, c1)과 (a2, b2, c2)의 외적은 교차점의 동차 좌표를 제공합니다.
    """
    # l1 = (a1, b1, c1), l2 = (a2, b2, c2)
    # 교차점 p = l1 x l2
    
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    x = b1*c2 - c1*b2
    y = c1*a2 - a1*c2
    w = a1*b2 - b1*a2
    
    if abs(w) < 1e-6:
        return None # 평행선
        
    return (x/w, y/w)

def ransac_vanishing_point(lines, num_iterations=1000, threshold=2.0):
    """
    RANSAC을 사용하여 하나의 지배적인 소실점을 찾습니다.
    
    인자:
        lines: (N, 4) 배열
        num_iterations: RANSAC 반복 횟수
        threshold: 인라이어에 대한 픽셀 단위 거리 임계값
        
    반환:
        best_vp: 최적의 소실점 (x, y)
        inliers: 인라이어의 불리언 마스크
    """
    if lines is None or len(lines) < 2:
        return None, np.array([])
        
    N = lines.shape[0]
    line_params = compute_line_params(lines) # (N, 3)
    
    best_score = -1
    best_vp = None
    best_inliers = np.zeros(N, dtype=bool)
    
    # 적응형 RANSAC 변수
    # log(1-p) / log(1-w^2) 여기서 w는 인라이어 비율. 높은 반복 횟수로 시작
    
    # 결정론적 동작을 위해 고정 시드 사용
    rng = np.random.default_rng(RANSAC_RANDOM_SEED)
    
    for _ in range(num_iterations):
        # 임의로 2개의 선 샘플링
        idx = rng.choice(N, 2, replace=False)
        l1 = line_params[idx[0]]
        l2 = line_params[idx[1]]
        
        # 교차점 계산
        vp = get_intersection(l1, l2)
        if vp is None:
            continue
            
        vx, vy = vp
        
        # VP가 매우 멀리 있는지 확인 (본질적으로 평행선)
        # 이를 무한대로 취급할 수 있지만, 시각화를 위해 제한하거나 다르게 처리할 수 있음
        # 일단 유지
        
        # 이 VP까지의 모든 선의 거리 계산
        # 점 (vx, vy)에서 직선 ax+by+c=0까지의 거리는 |a*vx + b*vy + c| (a,b가 정규화되었으므로)
        distances = np.abs(line_params[:, 0] * vx + line_params[:, 1] * vy + line_params[:, 2])
        
        inliers = (distances < threshold)
        score = np.sum(inliers)
        
        # 긴 선으로 형성된 VP를 선호하도록 선 길이에 가중치를 줄 수 있음
        # 하지만 현재는 단순 카운트만으로도 충분히 견고함
        
        if score > best_score:
            best_score = score
            best_vp = vp
            best_inliers = inliers
            
    # 모든 인라이어를 사용하여 최소 제곱법으로 VP 정제
    if best_vp is not None and np.sum(best_inliers) > 2:
        # 많은 선의 교차점을 찾기 위해 A x + B y + C = 0 -> A x + B y = -C
        # Ax = b를 사용하여 [x, y] 풀기
        A = line_params[best_inliers, :2]
        b = -line_params[best_inliers, 2]
        try:
            res = np.linalg.lstsq(A, b, rcond=None)
            best_vp = res[0]
        except:
            pass # 정제 실패 시 원래 RANSAC 추정치 유지
            
    return best_vp, best_inliers

def compute_vanishing_points(lines, image_shape, num_vps=3):
    """
    계단식 RANSAC을 사용하여 최대 num_vps개의 소실점을 찾습니다.
    """
    if lines is None or len(lines) < 2:
        return [], []
    
    N = lines.shape[0]
    remaining_indices = np.arange(N)
    
    vps = []
    vps_inlier_indices = []
    
    # 최대 num_vps까지 계산
    current_lines = lines
    current_indices = remaining_indices
    
    for i in range(num_vps):
        if len(current_lines) < 2:
            break
            
        vp, inliers_mask = ransac_vanishing_point(current_lines, num_iterations=500, threshold=3.0)
        
        if vp is None or np.sum(inliers_mask) < 5: # VP를 지원하는 최소 5개의 선
            break
            
        # 결과 저장
        # 참고: inliers_mask는 current_lines에 상대적임
        # 원래 인덱스로 다시 매핑해야 함
        global_inlier_indices = current_indices[inliers_mask]
        
        vps.append(vp)
        vps_inlier_indices.append(global_inlier_indices)
        
        # 다음 반복을 위해 인라이어 제거
        outliers_mask = ~inliers_mask
        current_lines = current_lines[outliers_mask]
        current_indices = current_indices[outliers_mask]
        
    return vps, vps_inlier_indices

def visualize_vanishing_points(image, lines, vps, vps_inliers):
    """
    이미지에 선과 VP를 그립니다.
    colors: vps [0], [1], [2]... 에 해당
    """
    vis_img = image.copy()
    h, w = image.shape[:2]
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR: Red, Green, Blue
    
    # 1. 모든 선을 먼저 회색으로 그리기 (희미하게)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(vis_img, (x1, y1), (x2, y2), (50, 50, 50), 1)
            
    # 2. 인라이어 선을 색상으로 그리기
    used_indices = set()
    for i, indices in enumerate(vps_inliers):
        color = colors[i % len(colors)]
        for idx in indices:
            used_indices.add(idx)
            x1, y1, x2, y2 = map(int, lines[idx])
            cv2.line(vis_img, (x1, y1), (x2, y2), color, 2)
            
    # 3. 소실점 그리기
    for i, vp in enumerate(vps):
        vx, vy = vp
        if np.abs(vx) > 10000 or np.abs(vy) > 10000:
            continue # 너무 멀어서 그릴 수 없음
            
        ix, iy = int(vx), int(vy)
        color = colors[i % len(colors)]
        
        # 이미지 중심에서 VP까지 찾는 선 그리기 (선택 사항, 산만할 수 있음)
        # cv2.line(vis_img, (w//2, h//2), (ix, iy), color, 1, cv2.LINE_AA)
        
        # VP 마커 그리기
        cv2.circle(vis_img, (ix, iy), 8, color, -1)
        cv2.circle(vis_img, (ix, iy), 10, (255, 255, 255), 2)
        
    return vis_img
