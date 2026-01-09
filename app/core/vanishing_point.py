import cv2
import numpy as np
import math

def detect_lines_lsd(image):
    """
    Detect line segments in the image using LSD (Line Segment Detector).
    
    Args:
        image: Input image (BGR or Grayscale).
        
    Returns:
        lines: A numpy array of shape (N, 4) containing line segments [x1, y1, x2, y2].
               Returns None if no lines are found.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Create LSD detector with default parameters
    # LSD_REFINE_STD is standard refinement
    lsd = cv2.createLineSegmentDetector(0) 
    
    lines, _, _, _ = lsd.detect(gray)
    
    if lines is not None:
        # lines format from LSD is (N, 1, 4), reshape to (N, 4)
        lines = lines.reshape(-1, 4)
        # Filter out very short lines (optional, but good for robustness)
        min_len = 15  # Minimum length in pixels
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
    Convert lines [x1, y1, x2, y2] to general form ax + by + c = 0.
    Also returns length and midpoint for weighting.
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
    # general form: a*x + b*y + c = 0
    # line through (x1, y1) and (x2, y2)
    # (y - y1) / (y2 - y1) = (x - x1) / (x2 - x1)
    # (y - y1)(x2 - x1) = (x - x1)(y2 - y1)
    # y(x2 - x1) - y1(x2 - x1) = x(y2 - y1) - x1(y2 - y1)
    # (y1 - y2)x + (x2 - x1)y + x1(y2 - y1) - y1(x2 - x1) = 0
    # a = y1 - y2
    # b = x2 - x1
    # c = x1*y2 - x2*y1  (Wait, let's check sigms)
    # -x1y1 + x1y2 + x1y1 - x2y1 = x1y2 - x2y1
    # So c = x1*y2 - x2*y1
    
    c = x1 * y2 - x2 * y1
    
    # Normalize (a, b) to unit vector for distance calculation distance = |ax+by+c| / sqrt(a^2+b^2)
    norm = np.sqrt(a**2 + b**2)
    
    # Avoid division by zero for points (length 0 lines) - though filtered out
    norm = np.maximum(norm, 1e-6)
    
    a_norm = a / norm
    b_norm = b / norm
    c_norm = c / norm
    
    return np.column_stack((a_norm, b_norm, c_norm))

def get_intersection(line1, line2):
    """
    Find intersection of two lines given in (a, b, c) form where ax+by+c=0.
    Cross product of (a1, b1, c1) and (a2, b2, c2) gives homogenous coordinate of intersection.
    """
    # l1 = (a1, b1, c1), l2 = (a2, b2, c2)
    # intersection p = l1 x l2
    
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    
    x = b1*c2 - c1*b2
    y = c1*a2 - a1*c2
    w = a1*b2 - b1*a2
    
    if abs(w) < 1e-6:
        return None # Parallel lines
        
    return (x/w, y/w)

def ransac_vanishing_point(lines, num_iterations=1000, threshold=2.0):
    """
    Find one dominant vanishing point using RANSAC.
    
    Args:
        lines: (N, 4) array
        num_iterations: Number of RANSAC iterations
        threshold: Distance threshold in pixels for inliers
        
    Returns:
        best_vp: (x, y) of the best vanishing point
        inliers: Boolean mask of inliers
    """
    if lines is None or len(lines) < 2:
        return None, np.array([])
        
    N = lines.shape[0]
    line_params = compute_line_params(lines) # (N, 3)
    
    best_score = -1
    best_vp = None
    best_inliers = np.zeros(N, dtype=bool)
    
    # Adaptive RANSAC variables
    # log(1-p) / log(1-w^2) where w is inlier ratio. Start with high iter
    
    for _ in range(num_iterations):
        # Sample 2 lines randomly
        idx = np.random.choice(N, 2, replace=False)
        l1 = line_params[idx[0]]
        l2 = line_params[idx[1]]
        
        # Calculate intersection
        vp = get_intersection(l1, l2)
        if vp is None:
            continue
            
        vx, vy = vp
        
        # Check if VP is extremely far away (essentially parallel lines)
        # We can treat this as infinity, but for visualization we might cap it or handle differently
        # For now, let's keep it.
        
        # Calculate distances of all lines to this VP
        # Distance from point (vx, vy) to line ax+by+c=0 is |a*vx + b*vy + c| (since a,b normalized)
        distances = np.abs(line_params[:, 0] * vx + line_params[:, 1] * vy + line_params[:, 2])
        
        inliers = (distances < threshold)
        score = np.sum(inliers)
        
        # We can weight the score by line length to prefer VPs formed by long lines
        # But simple count is robust enough for now
        
        if score > best_score:
            best_score = score
            best_vp = vp
            best_inliers = inliers
            
    # Refine VP using least squares with all inliers
    if best_vp is not None and np.sum(best_inliers) > 2:
        # To find intersection of many lines A x + B y + C = 0 -> A x + B y = -C
        # Solve for [x, y] using Ax = b
        A = line_params[best_inliers, :2]
        b = -line_params[best_inliers, 2]
        try:
            res = np.linalg.lstsq(A, b, rcond=None)
            best_vp = res[0]
        except:
            pass # Keep original RANSAC estimate if refinement fails
            
    return best_vp, best_inliers

def compute_vanishing_points(lines, image_shape, num_vps=3):
    """
    Cascaded RANSAC to find up to num_vps vanishing points.
    """
    if lines is None or len(lines) < 2:
        return [], []
    
    N = lines.shape[0]
    remaining_indices = np.arange(N)
    
    vps = []
    vps_inlier_indices = []
    
    # We will compute up to num_vps
    current_lines = lines
    current_indices = remaining_indices
    
    for i in range(num_vps):
        if len(current_lines) < 2:
            break
            
        vp, inliers_mask = ransac_vanishing_point(current_lines, num_iterations=500, threshold=3.0)
        
        if vp is None or np.sum(inliers_mask) < 5: # Minimum 5 lines to support a VP
            break
            
        # Store result
        # Note: inliers_mask is relative to current_lines
        # We need to map back to original indices
        global_inlier_indices = current_indices[inliers_mask]
        
        vps.append(vp)
        vps_inlier_indices.append(global_inlier_indices)
        
        # Remove inliers for next iteration
        outliers_mask = ~inliers_mask
        current_lines = current_lines[outliers_mask]
        current_indices = current_indices[outliers_mask]
        
    return vps, vps_inlier_indices

def visualize_vanishing_points(image, lines, vps, vps_inliers):
    """
    Draw lines and VPs on the image.
    colors: correspond to vps [0], [1], [2]...
    """
    vis_img = image.copy()
    h, w = image.shape[:2]
    
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # BGR: Red, Green, Blue
    
    # 1. Draw all lines in gray first (faint)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = map(int, line)
            cv2.line(vis_img, (x1, y1), (x2, y2), (50, 50, 50), 1)
            
    # 2. Draw inlier lines with colors
    used_indices = set()
    for i, indices in enumerate(vps_inliers):
        color = colors[i % len(colors)]
        for idx in indices:
            used_indices.add(idx)
            x1, y1, x2, y2 = map(int, lines[idx])
            cv2.line(vis_img, (x1, y1), (x2, y2), color, 2)
            
    # 3. Draw Vanishing Points
    for i, vp in enumerate(vps):
        vx, vy = vp
        if np.abs(vx) > 10000 or np.abs(vy) > 10000:
            continue # Too far to draw
            
        ix, iy = int(vx), int(vy)
        color = colors[i % len(colors)]
        
        # Draw finding lines from center of image to VP (optional, maybe distracting)
        # cv2.line(vis_img, (w//2, h//2), (ix, iy), color, 1, cv2.LINE_AA)
        
        # Draw VP marker
        cv2.circle(vis_img, (ix, iy), 8, color, -1)
        cv2.circle(vis_img, (ix, iy), 10, (255, 255, 255), 2)
        
    return vis_img
