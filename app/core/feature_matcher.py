"""
Feature matching and pose estimation utilities using SIFT and Essential Matrix
"""
import cv2
import numpy as np
import random
from typing import Tuple, List, Optional, Dict


def compute_sift_matches(img1: np.ndarray, img2: np.ndarray,
                         ratio_thresh: float = 0.75) -> Tuple[List, List, List]:
    """
    Compute SIFT feature matches between two images with cross-check.

    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        ratio_thresh: Lowe's ratio test threshold (default 0.75)

    Returns:
        Tuple of (keypoints1, keypoints2, good_matches)
    """
    # SIFT Detector
    sift = cv2.SIFT_create()

    # Detect and Compute
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    # Matcher
    bf = cv2.BFMatcher()

    # 1. Match A -> B
    matches_12 = bf.knnMatch(des1, des2, k=2)
    good_12 = []
    for m, n in matches_12:
        if m.distance < ratio_thresh * n.distance:
            good_12.append(m)

    # 2. Match B -> A (Cross Check)
    matches_21 = bf.knnMatch(des2, des1, k=2)
    good_21 = []
    for m, n in matches_21:
        if m.distance < ratio_thresh * n.distance:
            good_21.append(m)

    # 3. Find intersection (Mutual Matches)
    good = []
    matches_21_map = {m.queryIdx: m.trainIdx for m in good_21}

    for m in good_12:
        # m.queryIdx is A, m.trainIdx is B
        # Check if B(m.trainIdx) maps back to A(m.queryIdx)
        if m.trainIdx in matches_21_map:
            if matches_21_map[m.trainIdx] == m.queryIdx:
                good.append(m)

    return kp1, kp2, good


def compute_orb_matches(img1: np.ndarray, img2: np.ndarray,
                        ratio_thresh: float = 0.75) -> Tuple[List, List, List]:
    """
    Compute ORB feature matches between two images with cross-check.

    Args:
        img1: First image (numpy array)
        img2: Second image (numpy array)
        ratio_thresh: Lowe's ratio test threshold (default 0.75)

    Returns:
        Tuple of (keypoints1, keypoints2, good_matches)
    """
    # ORB Detector
    orb = cv2.ORB_create(nfeatures=2000)

    # Detect and Compute
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None:
        return kp1, kp2, []

    # Matcher (use NORM_HAMMING for ORB binary descriptors)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    # 1. Match A -> B
    matches_12 = bf.knnMatch(des1, des2, k=2)
    good_12 = []
    for pair in matches_12:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_12.append(m)

    # 2. Match B -> A (Cross Check)
    matches_21 = bf.knnMatch(des2, des1, k=2)
    good_21 = []
    for pair in matches_21:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good_21.append(m)

    # 3. Find intersection (Mutual Matches)
    good = []
    matches_21_map = {m.queryIdx: m.trainIdx for m in good_21}

    for m in good_12:
        # m.queryIdx is A, m.trainIdx is B
        # Check if B(m.trainIdx) maps back to A(m.queryIdx)
        if m.trainIdx in matches_21_map:
            if matches_21_map[m.trainIdx] == m.queryIdx:
                good.append(m)

    return kp1, kp2, good


def estimate_relative_pose(kp1, kp2, good_matches, camera_params: Dict,
                           img_shape: Tuple[int, int],
                           max_matches: int = 50) -> Tuple[Optional[np.ndarray],
                                                            Optional[np.ndarray],
                                                            List]:
    """
    Estimate relative pose (R, t) between two camera views using Essential Matrix.

    Args:
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        good_matches: List of good matches
        camera_params: Dictionary containing camera intrinsic parameters
        img_shape: Image shape (height, width)
        max_matches: Maximum number of matches to use (default 50)

    Returns:
        Tuple of (R_relative, t_relative, inlier_matches)
    """
    if len(good_matches) < 5:
        return None, None, []

    # Sort by match distance and take top N
    good_sorted = sorted(good_matches, key=lambda x: x.distance)[:max_matches]

    # Extract Points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_sorted]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_sorted]).reshape(-1, 1, 2)

    # Get Camera Matrix K & Distortion Coeffs
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

    # Undistort Points to Normalized Coordinates
    src_norm = cv2.undistortPoints(src_pts, K, dist_coeffs)
    dst_norm = cv2.undistortPoints(dst_pts, K, dist_coeffs)

    # Find Essential Matrix (Robust RANSAC)
    E, mask = cv2.findEssentialMat(
        src_norm, dst_norm,
        focal=1.0, pp=(0, 0),
        method=cv2.RANSAC, prob=0.999, threshold=0.001
    )

    if E is None:
        return None, None, []

    # Recover Pose (R, t)
    _, R_rel, t_rel, mask_pose = cv2.recoverPose(E, src_norm, dst_norm)

    # Filter inliers for display
    inliers = []
    if mask_pose is not None:
        matchesMask = mask_pose.ravel().tolist()
        inliers = [m for i, m in enumerate(good_sorted) if matchesMask[i]]
    else:
        inliers = good_sorted

    return R_rel, t_rel, inliers


def visualize_matches(img1: np.ndarray, img2: np.ndarray,
                      kp1, kp2, matches: List,
                      target_width: int = 800) -> np.ndarray:
    """
    Visualize feature matches between two images side by side.

    Args:
        img1: First image
        img2: Second image
        kp1: Keypoints from image 1
        kp2: Keypoints from image 2
        matches: List of matches to visualize
        target_width: Target width for each image (default 800)

    Returns:
        Visualization image with matches drawn
    """
    def resize_img(im):
        h, w = im.shape[:2]
        scale = target_width / w
        return cv2.resize(im, (target_width, int(h * scale)))

    img1_disp = resize_img(img1)
    img2_disp = resize_img(img2)

    # Canvas
    h1, w1 = img1_disp.shape[:2]
    h2, w2 = img2_disp.shape[:2]
    vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1_disp
    vis[:h2, w1:w1+w2] = img2_disp

    # Scaling factor for points
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
