"""
두 이미지를 비교하고 포즈를 추정하기 위한 특징 매칭 대화상자
"""
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QGroupBox, QScrollArea, QWidget
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from u1gis_geovision import CameraParams

from app.core.geometry import euler_to_R_cam, R_cam_to_euler
from app.core.feature_matcher import compute_sift_matches, estimate_relative_pose, visualize_matches


class MatcherDialog(QDialog):
    def __init__(self, img_path1, img_path2, parent=None, projector=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Comparison & Pose Estimation")
        self.resize(1600, 900)

        self.img_path1 = img_path1
        self.img_path2 = img_path2
        self.projector = projector  # 지도 투영용

        # 데이터 플레이스홀더
        self.params_A = None
        self.params_B = None # 계산됨
        self.R_relative = None
        self.t_relative = None

        # --- 레이아웃 ---
        main_layout = QVBoxLayout(self)

        # 1. 상단: 투영 지도 비교 (좌: 기존 파라미터, 우: 계산된 파라미터)
        projection_layout = QHBoxLayout()

        # 왼쪽: 기존 파라미터로 투영된 지도
        left_container = QVBoxLayout()
        self.lbl_left_title = QLabel("기존 파라미터 (A)")
        self.lbl_left_title.setAlignment(Qt.AlignCenter)
        self.lbl_left_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        left_container.addWidget(self.lbl_left_title)

        self.lbl_left_image = QLabel("이미지 로딩 중...")
        self.lbl_left_image.setAlignment(Qt.AlignCenter)
        self.lbl_left_image.setMinimumSize(880, 495)
        left_container.addWidget(self.lbl_left_image)
        projection_layout.addLayout(left_container)

        # 오른쪽: 계산된 파라미터로 투영된 지도
        right_container = QVBoxLayout()
        self.lbl_right_title = QLabel("계산된 파라미터 (B)")
        self.lbl_right_title.setAlignment(Qt.AlignCenter)
        self.lbl_right_title.setStyleSheet("font-weight: bold; font-size: 14px; color: blue;")
        right_container.addWidget(self.lbl_right_title)

        self.lbl_right_image = QLabel("계산 대기 중...")
        self.lbl_right_image.setAlignment(Qt.AlignCenter)
        self.lbl_right_image.setMinimumSize(880, 495)
        right_container.addWidget(self.lbl_right_image)
        projection_layout.addLayout(right_container)

        main_layout.addLayout(projection_layout, 7)

        # 2. 하단: 파라미터 제어
        self.controls_layout = QHBoxLayout()
        main_layout.addLayout(self.controls_layout, 3)

        # 그룹 A (참조)
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

        # 그룹 B (대상)
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

        # 저장 버튼
        self.btn_save = QPushButton("Save B Parameters to YAML")
        self.btn_save.setFixedHeight(40)
        self.btn_save.clicked.connect(self._save_B_yaml)
        # 하단의 B 레이아웃에 버튼 추가
        self.layout_B.addWidget(self.btn_save, 6, 0, 1, 2)

        self.controls_layout.addWidget(self.group_B)

        # 닫기 버튼
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        # main_layout.addWidget(btn_close) # 선택 사항, X 버튼으로 충분함

        # A의 YAML 확인
        self._load_A_params()

        # 매칭 및 추정 실행
        self._compute_and_show()

    def _get_camera_path(self, img_path_str):
        """
        이미지 경로를 해당 카메라 파라미터 경로로 변환합니다.
        .../image/Sequence/001.png -> .../camera/Sequence/001.yaml
        """
        img_path = Path(img_path_str)
        try:
            # 1. 단순한 'image' 폴더 구조에 있는지 확인
            # 경로 부분 분할
            parts = list(img_path.parts)
            # 'image' 부분 찾기 (문제를 피하기 위해 오른쪽에서 왼쪽으로?)
            # 표준 구조 가정: workspace/image/Sequence/file.png
            if 'image' in parts:
                idx = parts.index('image')
                # 마지막 부분(파일 이름)이 아닌지 확인
                if idx < len(parts) - 1:
                    parts[idx] = 'camera'
                    cam_path = Path(*parts).with_suffix('.yaml')
                    return cam_path
        except Exception:
            pass

        # 대체: 단순 구조인 경우 형제 디렉토리 "camera"
        # .../image/file.png -> .../camera/file.yaml (드묾)
        return img_path.parent / f"{img_path.stem}.yaml"

    def _load_A_params(self):
        """이미지 1에 대한 동반 YAML 로드 시도"""
        yaml_path = self._get_camera_path(self.img_path1)

        # 1. 특정 파일 시도 (예: 001.yaml)
        if not yaml_path.exists():
            # 2. 같은 디렉토리의 base.yaml 시도
            base_path = yaml_path.parent / "base.yaml"
            if base_path.exists():
                yaml_path = base_path
            else:
                # 3. 최후의 수단: yaml이 이미지 바로 옆에 있는지 확인 (레거시/플랫)
                flat_path = Path(self.img_path1).with_suffix('.yaml')
                if flat_path.exists():
                    yaml_path = flat_path

        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data = yaml.safe_load(f)
                self.params_A = data # 딕셔너리

                # UI A 업데이트
                self.lbls_A["X"].setText(f"{data.get('x', 0):.3f}")
                self.lbls_A["Y"].setText(f"{data.get('y', 0):.3f}")
                self.lbls_A["Z"].setText(f"{data.get('z', 0):.3f}")
                self.lbls_A["Yaw"].setText(f"{data.get('yaw', 0):.3f}")
                self.lbls_A["Pitch"].setText(f"{data.get('pitch', 0):.3f}")
                self.lbls_A["Roll"].setText(f"{data.get('roll', 0):.3f}")

            except Exception as e:
                print(f"A yaml 로드 실패: {e}")

    def _save_B_yaml(self):
        """계산된 B 파라미터를 YAML로 저장"""
        if self.params_B is None:
            return

        # 저장 경로 해결
        save_path = self._get_camera_path(self.img_path2)

        # 디렉토리가 존재하는지 확인 ('camera' 폴더가 없는 경우)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_path, 'w') as f:
                yaml.dump(self.params_B, f, default_flow_style=False, sort_keys=False)

            # 성공 메시지 표시 (제목 또는 레이블의 간단한 업데이트)
            self.setWindowTitle(f"Saved to {save_path.name}!")
            self.btn_save.setText("Saved!")
            self.btn_save.setEnabled(False)
        except Exception as e:
            print(f"저장 실패: {e}")

    def _compute_and_show(self):
        img1 = cv2.imread(self.img_path1)
        img2 = cv2.imread(self.img_path2)

        if img1 is None or img2 is None:
            self.lbl_left_image.setText("이미지 로드 실패")
            self.lbl_right_image.setText("이미지 로드 실패")
            return

        # 교차 검사를 통한 SIFT 매칭 계산
        kp1, kp2, good_matches = compute_sift_matches(img1, img2, ratio_thresh=0.75)

        if len(good_matches) == 0:
            self.lbl_left_image.setText("특징점을 찾을 수 없습니다.")
            self.lbl_right_image.setText("특징점을 찾을 수 없습니다.")
            return

        # --- 포즈 추정 로직 ---
        inliers = []
        self.R_relative = np.eye(3)
        self.params_B = None

        if self.params_A and len(good_matches) >= 5:
            # 상대 포즈 추정
            R_rel, t_rel, inliers = estimate_relative_pose(
                kp1, kp2, good_matches, self.params_A, img1.shape, max_matches=50
            )

            if R_rel is not None:
                self.R_relative = R_rel

                # B의 절대 회전 계산
                # R_cam_b = R_rel @ R_cam_a
                yaw_a = self.params_A.get('yaw', 0)
                pitch_a = self.params_A.get('pitch', 0)
                roll_a = self.params_A.get('roll', 0)

                R_a = euler_to_R_cam(yaw_a, pitch_a, roll_a)
                R_b = R_rel @ R_a

                # 오일러로 다시 변환
                yaw_b, pitch_b, roll_b = R_cam_to_euler(R_b)

                # 파라미터 B 설정
                self.params_B = self.params_A.copy()
                self.params_B['yaw'] = float(yaw_b)
                self.params_B['pitch'] = float(pitch_b)
                self.params_B['roll'] = float(roll_b)

                # UI 업데이트
                self.lbls_B["X"].setText(f"{self.params_B['x']:.3f} (=)")
                self.lbls_B["Y"].setText(f"{self.params_B['y']:.3f} (=)")
                self.lbls_B["Z"].setText(f"{self.params_B['z']:.3f} (=)")
                self.lbls_B["Yaw"].setText(f"{yaw_b:.3f}")
                self.lbls_B["Pitch"].setText(f"{pitch_b:.3f}")
                self.lbls_B["Roll"].setText(f"{roll_b:.3f}")
            else:
                self.lbl_left_image.setText("Essential Matrix 계산 실패")
                self.lbl_right_image.setText("Essential Matrix 계산 실패")
                return
        else:
            self.lbl_left_image.setText("매칭점이 부족합니다 (< 5).")
            self.lbl_right_image.setText("매칭점이 부족합니다 (< 5).")
            inliers = good_matches  # 포즈 추정이 없는 경우 모든 매칭 사용

        # --- 투영 지도 시각화 ---
        self._show_projected_maps(img1, img2, kp1, kp2, inliers)

    def _show_projected_maps(self, img1, img2, kp1, kp2, inliers):
        """왼쪽에 기존 파라미터, 오른쪽에 계산된 파라미터로 투영된 지도 표시"""
        target_width = 880

        # inlier 특징점 좌표와 색상 추출
        pts1 = []
        pts2 = []
        colors = []
        np.random.seed(42)  # 재현성을 위한 시드
        for m in inliers:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
            # 각 매칭 쌍마다 랜덤 색상 생성
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)

        # 왼쪽: 기존 파라미터 (A)로 투영 + 특징점
        left_img = self._draw_projection_overlay(img1, self.params_A, target_width, pts1, colors)
        self._display_image(self.lbl_left_image, left_img)

        # 오른쪽: 계산된 파라미터 (B)로 투영 + 특징점
        if self.params_B is not None:
            right_img = self._draw_projection_overlay(img2, self.params_B, target_width, pts2, colors)
            self._display_image(self.lbl_right_image, right_img)
        else:
            # params_B가 없으면 원본 이미지 + 특징점만 표시
            right_img = self._draw_projection_overlay(img2, None, target_width, pts2, colors)
            self._display_image(self.lbl_right_image, right_img)

    def _draw_projection_overlay(self, image, params_dict, target_width, keypoints=None, colors=None):
        """이미지 위에 투영된 지도 선과 특징점 그리기"""
        # 이미지 복사
        result = image.copy()

        # projector가 있고 params가 있는 경우에만 투영
        if self.projector is not None and params_dict is not None:
            try:
                # 딕셔너리에서 CameraParams 생성
                camera_params = CameraParams(**params_dict)

                # 지도 투영
                projected_data = self.projector.projection(camera_params).projected_layers

                # b2 레이어 (차선) 그리기
                if projected_data and projected_data.get('b2'):
                    for line in projected_data['b2']:
                        if len(line) > 0:
                            line_arr = np.array(line, dtype=np.int32)
                            cv2.polylines(result, [line_arr], isClosed=False, color=(255, 0, 0), thickness=2)
            except Exception as e:
                print(f"[MatcherDialog] 투영 오류: {e}")

        # 특징점 그리기
        if keypoints:
            for i, pt in enumerate(keypoints):
                x, y = int(pt[0]), int(pt[1])
                color = colors[i] if colors and i < len(colors) else (0, 255, 0)
                cv2.circle(result, (x, y), 10, (255, 255, 255), 2)  # 흰색 외곽
                cv2.circle(result, (x, y), 8, (0, 0, 0), 2)         # 검은 테두리
                cv2.circle(result, (x, y), 6, color, -1)            # 색상 원

        # 리사이즈
        return self._resize_image(result, target_width)

    def _resize_image(self, image, target_width):
        """이미지를 지정된 너비로 리사이즈"""
        h, w = image.shape[:2]
        scale = target_width / w
        new_h = int(h * scale)
        return cv2.resize(image, (target_width, new_h))

    def _display_image(self, label, image):
        """QLabel에 OpenCV 이미지 표시"""
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)
