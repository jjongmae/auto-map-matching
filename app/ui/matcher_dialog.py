"""
Feature matching dialog for comparing two images and estimating pose
"""
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from app.core.geometry import euler_to_R_cam, R_cam_to_euler
from app.core.feature_matcher import compute_sift_matches, estimate_relative_pose, visualize_matches


class MatcherDialog(QDialog):
    def __init__(self, img_path1, img_path2, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Comparison & Pose Estimation")
        self.resize(1600, 900)

        self.img_path1 = img_path1
        self.img_path2 = img_path2

        # Data Placeholders
        self.params_A = None
        self.params_B = None # Calculated
        self.R_relative = None
        self.t_relative = None

        # --- Layout ---
        main_layout = QVBoxLayout(self)

        # 1. Top: Image Display (Side by Side)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.lbl_result = QLabel("Computing matches...")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        scroll.setWidget(self.lbl_result)
        main_layout.addWidget(scroll, 7)

        # 2. Bottom: Parameters Control
        self.controls_layout = QHBoxLayout()
        main_layout.addLayout(self.controls_layout, 3)

        # Group A (Reference)
        self.group_A = QGroupBox("Reference Camera A (Read-Only)")
        self.layout_A = QGridLayout()
        self.group_A.setLayout(self.layout_A)
        self.lbls_A = {}
        for i, key in enumerate(["X", "Y", "Z", "Yaw", "Pitch", "Roll"]):
            self.layout_A.addWidget(QLabel(key), i, 0)
            lbl_val = QLabel("-")
            self.lbls_A[key] = lbl_val
            self.layout_A.addWidget(lbl_val, i, 1)
        self.controls_layout.addWidget(self.group_A)

        # Group B (Target)
        self.group_B = QGroupBox("Target Camera B (Read-Only / Calculated)")
        self.layout_B = QGridLayout()
        self.group_B.setLayout(self.layout_B)
        self.lbls_B = {}
        for i, key in enumerate(["X", "Y", "Z", "Yaw", "Pitch", "Roll"]):
            self.layout_B.addWidget(QLabel(key), i, 0)
            lbl_val = QLabel("-")
            lbl_val.setStyleSheet("font-weight: bold; color: blue;")
            self.lbls_B[key] = lbl_val
            self.layout_B.addWidget(lbl_val, i, 1)

        # Save Button
        self.btn_save = QPushButton("Save B Parameters to YAML")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self._save_B_yaml)
        # Add button to B's layout at the bottom
        self.layout_B.addWidget(self.btn_save, 6, 0, 1, 2)

        self.controls_layout.addWidget(self.group_B)

        # Close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        # main_layout.addWidget(btn_close) # Optional, X button is enough

        # Check for A's YAML
        self._load_A_params()

        # Run matching & estimation
        self._compute_and_show()

    def _get_camera_path(self, img_path_str):
        """
        Convert image path to corresponding camera parameter path.
        .../image/Sequence/001.png -> .../camera/Sequence/001.yaml
        """
        img_path = Path(img_path_str)
        try:
            # 1. Check if we are in a simplistic 'image' folder structure
            # Split path parts
            parts = list(img_path.parts)
            # Find the 'image' part (from right to left to avoid issues?)
            # Assuming standard structure: workspace/image/Sequence/file.png
            if 'image' in parts:
                idx = parts.index('image')
                # Ensure it's not the last part (file name)
                if idx < len(parts) - 1:
                    parts[idx] = 'camera'
                    cam_path = Path(*parts).with_suffix('.yaml')
                    return cam_path
        except Exception:
            pass

        # Fallback: sibling directory "camera" if simple structure
        # .../image/file.png -> .../camera/file.yaml (rare)
        return img_path.parent / f"{img_path.stem}.yaml"

    def _load_A_params(self):
        """Try to load accompanying yaml for image 1"""
        yaml_path = self._get_camera_path(self.img_path1)

        # 1. Try specific file (e.g. 001.yaml)
        if not yaml_path.exists():
            # 2. Try base.yaml in the same directory
            base_path = yaml_path.parent / "base.yaml"
            if base_path.exists():
                yaml_path = base_path
            else:
                # 3. Last resort: check if yaml is just next to the image (legacy/flat)
                flat_path = Path(self.img_path1).with_suffix('.yaml')
                if flat_path.exists():
                    yaml_path = flat_path

        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                self.params_A = data # Dict

                # Update UI A
                self.lbls_A["X"].setText(f"{data.get('x', 0):.3f}")
                self.lbls_A["Y"].setText(f"{data.get('y', 0):.3f}")
                self.lbls_A["Z"].setText(f"{data.get('z', 0):.3f}")
                self.lbls_A["Yaw"].setText(f"{data.get('yaw', 0):.3f}")
                self.lbls_A["Pitch"].setText(f"{data.get('pitch', 0):.3f}")
                self.lbls_A["Roll"].setText(f"{data.get('roll', 0):.3f}")

            except Exception as e:
                print(f"Failed to load A yaml: {e}")

    def _save_B_yaml(self):
        """Save calculated B params to YAML"""
        if self.params_B is None:
            return

        # Resolve save path
        save_path = self._get_camera_path(self.img_path2)

        # Ensure directory exists (in case 'camera' folder is missing)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.params_B, f, default_flow_style=False, sort_keys=False)

            # Show success message (simple update to title or label)
            self.setWindowTitle(f"Saved to {save_path.name}!")
            self.btn_save.setText("Saved!")
            self.btn_save.setEnabled(False)
        except Exception as e:
            print(f"Save failed: {e}")

    def _compute_and_show(self):
        img1 = cv2.imread(self.img_path1)
        img2 = cv2.imread(self.img_path2)

        if img1 is None or img2 is None:
            self.lbl_result.setText("이미지 로드 실패")
            return

        # Compute SIFT matches with cross-check
        kp1, kp2, good_matches = compute_sift_matches(img1, img2, ratio_thresh=0.75)

        if len(good_matches) == 0:
            self.lbl_result.setText("특징점을 찾을 수 없습니다.")
            return

        # --- Pose Estimation Logic ---
        inliers = []
        self.R_relative = np.eye(3)
        self.params_B = None

        if self.params_A and len(good_matches) >= 5:
            # Estimate relative pose
            R_rel, t_rel, inliers = estimate_relative_pose(
                kp1, kp2, good_matches, self.params_A, img1.shape, max_matches=50
            )

            if R_rel is not None:
                self.R_relative = R_rel

                # Calculate B's absolute Rotation
                # R_cam_b = R_rel @ R_cam_a
                yaw_a = self.params_A.get('yaw', 0)
                pitch_a = self.params_A.get('pitch', 0)
                roll_a = self.params_A.get('roll', 0)

                R_a = euler_to_R_cam(yaw_a, pitch_a, roll_a)
                R_b = R_rel @ R_a

                # Convert back to Euler
                yaw_b, pitch_b, roll_b = R_cam_to_euler(R_b)

                # Set Params B
                self.params_B = self.params_A.copy()
                self.params_B['yaw'] = float(yaw_b)
                self.params_B['pitch'] = float(pitch_b)
                self.params_B['roll'] = float(roll_b)

                # Update UI
                self.lbls_B["X"].setText(f"{self.params_B['x']:.3f} (=)")
                self.lbls_B["Y"].setText(f"{self.params_B['y']:.3f} (=)")
                self.lbls_B["Z"].setText(f"{self.params_B['z']:.3f} (=)")
                self.lbls_B["Yaw"].setText(f"{yaw_b:.3f}")
                self.lbls_B["Pitch"].setText(f"{pitch_b:.3f}")
                self.lbls_B["Roll"].setText(f"{roll_b:.3f}")
            else:
                self.lbl_result.setText("Essential Matrix 계산 실패")
                return
        else:
            self.lbl_result.setText("매칭점이 부족합니다 (< 5).")
            inliers = good_matches  # Use all matches if no pose estimation

        # --- Visualization ---
        vis = visualize_matches(img1, img2, kp1, kp2, inliers, target_width=800)

        # Display
        h, w, ch = vis.shape
        bytes_per_line = ch * w
        qt_image = QImage(vis.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.lbl_result.setPixmap(pixmap)
