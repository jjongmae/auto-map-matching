"""
SIFT 및 Essential Matrix를 사용한 특징 매칭 및 포즈 추정 유틸리티
"""
import cv2
import numpy as np
import random
from typing import Tuple, List, Optional, Dict


def compute_sift_matches(img1: np.ndarray, img2: np.ndarray,
                         ratio_thresh: float = 0.75) -> Tuple[List, List, List]:
    """
    교차 검사를 통해 두 이미지 간의 SIFT 특징 매칭을 계산합니다.

    인자:
        img1: 첫 번째 이미지 (numpy 배열)
        img2: 두 번째 이미지 (numpy 배열)
        ratio_thresh: Lowe의 비율 테스트 임계값 (기본값 0.75)

    반환:
        (keypoints1, keypoints2, good_matches)의 튜플
    """
    # SIFT 검출기
    sift = cv2.SIFT_create()

    # 검출 및 계산
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    print(f"[SIFT] 이미지1 키포인트: {len(kp1)}개 검출")
    print(f"[SIFT] 이미지2 키포인트: {len(kp2)}개 검출")

    if des1 is None or des2 is None:
        return kp1, kp2, []

    # 매처
    bf = cv2.BFMatcher()

    # 1. 매칭 A -> B
    matches_12 = bf.knnMatch(des1, des2, k=2)
    good_12 = []
    for m, n in matches_12:
        if m.distance < ratio_thresh * n.distance:
            good_12.append(m)

    # 2. 매칭 B -> A (교차 검사)
    matches_21 = bf.knnMatch(des2, des1, k=2)
    good_21 = []
    for m, n in matches_21:
        if m.distance < ratio_thresh * n.distance:
            good_21.append(m)

    # 3. 교차점 찾기 (상호 매칭)
    good = []
    matches_21_map = {m.queryIdx: m.trainIdx for m in good_21}

    print(f"[SIFT] 1차 매칭(A->B) 통과: {len(good_12)}개")
    
    for m in good_12:
        # m.queryIdx는 A, m.trainIdx는 B
        # B(m.trainIdx)가 A(m.queryIdx)로 다시 매핑되는지 확인
        if m.trainIdx in matches_21_map:
            if matches_21_map[m.trainIdx] == m.queryIdx:
                good.append(m)

    print(f"[SIFT] 최종 상호 매칭(Cross Check): {len(good)}개")

    return kp1, kp2, good


def compute_orb_matches(img1: np.ndarray, img2: np.ndarray,
                        ratio_thresh: float = 0.75) -> Tuple[List, List, List]:
    """
    교차 검사를 통해 두 이미지 간의 ORB 특징 매칭을 계산합니다.

    인자:
        img1: 첫 번째 이미지 (numpy 배열)
        img2: 두 번째 이미지 (numpy 배열)
        ratio_thresh: Lowe의 비율 테스트 임계값 (기본값 0.75)

    반환:
        (keypoints1, keypoints2, good_matches)의 튜플
    """
    # ORB 검출기
    orb = cv2.ORB_create(nfeatures=2000)

    # 검출 및 계산
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    print(f"[ORB] 이미지1 키포인트: {len(kp1)}개 검출")
    print(f"[ORB] 이미지2 키포인트: {len(kp2)}개 검출")

    if des1 is None or des2 is None:
        return kp1, kp2, []

    # 매처 (ORB 바이너리 디스크립터에 NORM_HAMMING 사용)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 1. 매칭 A -> B
    matches_12 = bf.knnMatch(des1, des2, k=2)
    good_12 = []
    for pair in matches_12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_12.append(m)

    # 2. 매칭 B -> A (교차 검사)
    matches_21 = bf.knnMatch(des2, des1, k=2)
    good_21 = []
    for pair in matches_21:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_21.append(m)

    # 3. 교차점 찾기 (상호 매칭)
    good = []
    matches_21_map = {m.queryIdx: m.trainIdx for m in good_21}

    print(f"[ORB] 1차 매칭(A->B) 통과: {len(good_12)}개")

    for m in good_12:
        # m.queryIdx는 A, m.trainIdx는 B
        # B(m.trainIdx)가 A(m.queryIdx)로 다시 매핑되는지 확인
        if m.trainIdx in matches_21_map:
            if matches_21_map[m.trainIdx] == m.queryIdx:
                good.append(m)

    print(f"[ORB] 최종 상호 매칭(Cross Check): {len(good)}개")

    return kp1, kp2, good


def estimate_relative_pose(kp1, kp2, good_matches, camera_params: Dict,
                           img_shape: Tuple[int, int],
                           max_matches: int = 50) -> Tuple[Optional[np.ndarray],
                                                            Optional[np.ndarray],
                                                            List]:
    """
    Essential Matrix를 사용하여 두 카메라 뷰 간의 상대 포즈(R, t)를 추정합니다.

    인자:
        kp1: 이미지 1의 키포인트
        kp2: 이미지 2의 키포인트
        good_matches: 좋은 매칭 목록
        camera_params: 카메라 내부 파라미터를 포함하는 딕셔너리
        img_shape: 이미지 모양 (height, width)
        max_matches: 사용할 최대 매칭 수 (기본값 50)

    반환:
        (R_relative, t_relative, inlier_matches)의 튜플
    """
    if len(good_matches) < 5:
        print(f"[Pose] 매칭 포인트 부족: {len(good_matches)}개 (최소 5개 필요)")
        return None, None, []

    print(f"[Pose] 포즈 추정 입력 매칭 수: {len(good_matches)}개")

    # 매칭 거리로 정렬하고 상위 N개 가져오기
    good_sorted = sorted(good_matches, key=lambda x: x.distance)[:max_matches]
    print(f"[Pose] 상위 {len(good_sorted)}개 매칭 포인트 사용 (최대 {max_matches}개 제한)")

    # 포인트 추출
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_sorted]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_sorted]).reshape(-1, 1, 2)

    # 카메라 행렬 K 및 왜곡 계수 가져오기
    fx = camera_params.get('fx', 1000)
    fy = camera_params.get('fy', 1000)
    cx = camera_params.get('cx', img_shape[1]/2)
    cy = camera_params.get('cy', img_shape[0]/2)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    dist_coeffs = np.array([
        camera_params.get('k1', 0), camera_params.get('k2', 0),
        camera_params.get('p1', 0), camera_params.get('p2', 0),
        camera_params.get('k3', 0)
    ], dtype=np.float64)

    # 정규화된 좌표로 포인트 왜곡 보정
    src_norm = cv2.undistortPoints(src_pts, K, dist_coeffs)
    dst_norm = cv2.undistortPoints(dst_pts, K, dist_coeffs)

    # Essential Matrix 찾기 (강건한 RANSAC)
    E, mask = cv2.findEssentialMat(
        src_norm, dst_norm,
        focal=1.0, pp=(0, 0),
        method=cv2.RANSAC, prob=0.999, threshold=0.001
    )

    if E is None or E.shape != (3, 3):
        print(f"[Pose] Essential Matrix 계산 실패 또는 유효하지 않은 형태")
        return None, None, []

    inliers_E = np.sum(mask) if mask is not None else 0
    print(f"[Pose] Essential Matrix 계산 성공 (RANSAC). 인라이어 수: {inliers_E}")

    # 포즈(R, t) 복원
    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, src_norm, dst_norm)

    inliers_pose = np.sum(mask_pose) if mask_pose is not None else 0
    print(f"[Pose] 포즈(R, t) 복원 완료. 최종 인라이어: {inliers_pose}")
    # print(f"[Pose] 추정된 평행이동 벡터(t):\n{t_rel.flatten()}")

    # 포즈 계산에 투입된 상위 매칭 점(good_sorted)을 모두 그대로 반환하여 시각화합니다.
    return R_rel, t_rel, good_sorted


def visualize_matches(img1: np.ndarray, img2: np.ndarray,
                      kp1, kp2, matches: List,
                      target_width: int = 800) -> np.ndarray:
    """
    두 이미지 간의 특징 매칭을 나란히 시각화합니다.

    인자:
        img1: 첫 번째 이미지
        img2: 두 번째 이미지
        kp1: 이미지 1의 키포인트
        kp2: 이미지 2의 키포인트
        matches: 시각화할 매칭 목록
        target_width: 각 이미지의 목표 너비 (기본값 800)

    반환:
        매칭이 그려진 시각화 이미지
    """
    def resize_img(im):
        h, w = im.shape[:2]
        scale = target_width / w
        return cv2.resize(im, (target_width, int(h * scale)))

    img1_disp = resize_img(img1)
    img2_disp = resize_img(img2)

    # 캔버스
    h1, w1 = img1_disp.shape[:2]
    h2, w2 = img2_disp.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1_disp
    vis[:h2, w1:w1+w2] = img2_disp

    # 포인트에 대한 스케일링 팩터
    s1 = target_width / img1.shape[1]
    s2 = target_width / img2.shape[1]

    for m in matches:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        pt1 = np.array(kp1[m.queryIdx].pt) * s1
        pt2 = np.array(kp2[m.trainIdx].pt) * s2

        pt1 = tuple(pt1.astype(int))
        pt2 = tuple(pt2.astype(int))
        pt2_shifted = (pt2[0] + w1, pt2[1])

        cv2.circle(vis, pt1, 6, color, -1)
        cv2.circle(vis, pt1, 6, (255, 255, 255), 1)
        cv2.circle(vis, pt2_shifted, 6, color, -1)
        cv2.circle(vis, pt2_shifted, 6, (255, 255, 255), 1)

    return vis
