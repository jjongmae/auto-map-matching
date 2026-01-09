"""
Main window for Map Matching application
"""
import os
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QDoubleSpinBox, QGroupBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QComboBox, QFileDialog, QListWidget, QCheckBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from vaid_gis import MapProjector, CameraParams


class MapMatcherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Matcher")
        self.setGeometry(100, 100, 1600, 900)

        # --- Data Attributes ---
        self.position_controls = {}
        self.orientation_controls = {}
        self.intrinsic_controls = {}
        self.current_image_path = None
        self.current_camera_yaml = None
        self.show_map_lines = True  # Default: show map lines

        # VAID GIS
        self.projector = None
        self.projected_map_data = None
        self.camera_params = CameraParams()

        # --- UI Setup ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # Main content: Image (left) + Controls (right)
        main_content = QHBoxLayout()
        root_layout.addLayout(main_content)

        # Left: Image Display
        image_panel = QVBoxLayout()
        self.image_label = QLabel("이미지를 선택하세요")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: grey; font-size: 20px;")
        self.image_label.setFixedSize(1280, 720)
        image_panel.addWidget(self.image_label)
        image_panel.addWidget(self._create_vision_group())
        main_content.addLayout(image_panel, 7)

        # Right: Controls
        controls_panel = QVBoxLayout()
        main_content.addLayout(controls_panel, 3)

        controls_panel.addStretch()
        controls_panel.addWidget(self._create_camera_selection_group())
        controls_panel.addWidget(self._create_image_selection_group())
        controls_panel.addWidget(self._create_position_group())
        controls_panel.addWidget(self._create_orientation_group())
        controls_panel.addWidget(self._create_intrinsic_group())
        controls_panel.addWidget(self._create_actions_group())
        controls_panel.addStretch()

        # Status Bar
        self.status_label = QLabel("준비 완료")
        self.status_label.setStyleSheet('''
            background-color: #F0F0F0;
            color: #333333;
            border: 1px solid #DCDCDC;
            border-radius: 5px;
            padding: 8px;
            font-size: 14px;
        ''')
        root_layout.addWidget(self.status_label)

        # Initial camera list population
        self._populate_camera_list()

    def _create_camera_selection_group(self):
        group = QGroupBox("카메라 선택 (YAML)")
        layout = QVBoxLayout()

        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_selected)

        layout.addWidget(self.camera_combo)
        group.setLayout(layout)

        return group

    def _create_image_selection_group(self):
        group = QGroupBox("이미지 선택")
        layout = QVBoxLayout()

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self._on_image_selected)

        layout.addWidget(self.image_list)
        group.setLayout(layout)
        return group

    def _create_position_group(self):
        group = QGroupBox("Position (좌표)")
        layout = QGridLayout()

        for i, name in enumerate(["X", "Y", "Z"]):
            slider = QSlider(Qt.Horizontal)
            spinbox = QDoubleSpinBox()
            self.position_controls[name] = {
                'slider': slider,
                'spinbox': spinbox,
                'base_value': 0
            }

            spinbox.setRange(-99999999, 99999999)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.1)

            slider_range = 100
            slider_scale = 100
            slider.setRange(-slider_range * slider_scale, slider_range * slider_scale)

            def make_spinbox_handler(ctrl, scale=slider_scale):
                def handler(val):
                    ctrl['base_value'] = val
                    ctrl['slider'].blockSignals(True)
                    ctrl['slider'].setValue(0)
                    ctrl['slider'].blockSignals(False)
                    self._update_projection()
                return handler

            def make_slider_handler(ctrl, scale=slider_scale):
                def handler(val):
                    new_val = ctrl['base_value'] + (val / scale)
                    ctrl['spinbox'].blockSignals(True)
                    ctrl['spinbox'].setValue(new_val)
                    ctrl['spinbox'].blockSignals(False)
                    self._update_projection()
                return handler

            spinbox.valueChanged.connect(make_spinbox_handler(self.position_controls[name]))
            slider.valueChanged.connect(make_slider_handler(self.position_controls[name]))

            layout.addWidget(QLabel(name), i, 0)
            layout.addWidget(slider, i, 1)
            layout.addWidget(spinbox, i, 2)

        group.setLayout(layout)
        return group

    def _create_orientation_group(self):
        group = QGroupBox("Orientation (자세)")
        layout = QGridLayout()

        ranges = {
            "Yaw": (-180, 180),
            "Pitch": (-90, 90),
            "Roll": (-180, 180)
        }

        for i, name in enumerate(["Yaw", "Pitch", "Roll"]):
            slider = QSlider(Qt.Horizontal)
            spinbox = QDoubleSpinBox()
            self.orientation_controls[name] = {
                'slider': slider,
                'spinbox': spinbox
            }

            min_val, max_val = ranges[name]
            spinbox.setRange(min_val, max_val)
            spinbox.setDecimals(3)
            spinbox.setSingleStep(0.1)

            slider_scale = 1000
            slider.setRange(min_val * slider_scale, max_val * slider_scale)

            def make_spinbox_handler(ctrl, scale=slider_scale):
                def handler(val):
                    ctrl['slider'].blockSignals(True)
                    ctrl['slider'].setValue(int(val * scale))
                    ctrl['slider'].blockSignals(False)
                    self._update_projection()
                return handler

            def make_slider_handler(ctrl, scale=slider_scale):
                def handler(val):
                    ctrl['spinbox'].blockSignals(True)
                    ctrl['spinbox'].setValue(val / scale)
                    ctrl['spinbox'].blockSignals(False)
                    self._update_projection()
                return handler

            spinbox.valueChanged.connect(make_spinbox_handler(self.orientation_controls[name]))
            slider.valueChanged.connect(make_slider_handler(self.orientation_controls[name]))

            layout.addWidget(QLabel(name), i, 0)
            layout.addWidget(slider, i, 1)
            layout.addWidget(spinbox, i, 2)

        group.setLayout(layout)
        return group

    def _create_intrinsic_group(self):
        group = QGroupBox("Intrinsic (내부 파라미터)")
        layout = QGridLayout()

        for i, name in enumerate(["fx", "fy"]):
            slider = QSlider(Qt.Horizontal)
            spinbox = QDoubleSpinBox()
            self.intrinsic_controls[name] = {
                'slider': slider,
                'spinbox': spinbox
            }

            spinbox.setRange(100, 5000)
            spinbox.setDecimals(2)
            spinbox.setSingleStep(1)

            slider_scale = 100
            slider.setRange(100 * slider_scale, 5000 * slider_scale)

            def make_spinbox_handler(ctrl, scale=slider_scale):
                def handler(val):
                    ctrl['slider'].blockSignals(True)
                    ctrl['slider'].setValue(int(val * scale))
                    ctrl['slider'].blockSignals(False)
                    self._update_projection()
                return handler

            def make_slider_handler(ctrl, scale=slider_scale):
                def handler(val):
                    ctrl['spinbox'].blockSignals(True)
                    ctrl['spinbox'].setValue(val / scale)
                    ctrl['spinbox'].blockSignals(False)
                    self._update_projection()
                return handler

            spinbox.valueChanged.connect(make_spinbox_handler(self.intrinsic_controls[name]))
            slider.valueChanged.connect(make_slider_handler(self.intrinsic_controls[name]))

            layout.addWidget(QLabel(name), i, 0)
            layout.addWidget(slider, i, 1)
            layout.addWidget(spinbox, i, 2)

        group.setLayout(layout)
        return group

    def _create_actions_group(self):
        group = QGroupBox("Actions")
        layout = QVBoxLayout()

        # Checkbox for showing/hiding map lines
        self.show_map_checkbox = QCheckBox("지도 선 표시")
        self.show_map_checkbox.setChecked(True)  # Default: checked
        self.show_map_checkbox.toggled.connect(self._on_map_visibility_changed)
        layout.addWidget(self.show_map_checkbox)

        # Buttons
        button_layout = QHBoxLayout()
        save_btn = QPushButton("Save")
        reset_btn = QPushButton("Reset")

        save_btn.clicked.connect(self._save_camera_yaml)
        reset_btn.clicked.connect(self._reset_to_yaml)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(reset_btn)
        layout.addLayout(button_layout)

        group.setLayout(layout)
        return group

    def _populate_camera_list(self):
        """Scan camera folder for subfolders containing base.yaml"""
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()

        camera_dir = Path("camera")
        if not camera_dir.exists():
            self.status_label.setText("camera 폴더가 없습니다")
            self.camera_combo.blockSignals(False)
            return

        # Find folders with base.yaml
        camera_folders = []
        for folder in camera_dir.iterdir():
            if folder.is_dir():
                base_yaml = folder / "base.yaml"
                if base_yaml.exists():
                    camera_folders.append(folder)

        if camera_folders:
            for folder in sorted(camera_folders):
                self.camera_combo.addItem(folder.name, userData=str(folder))
            self.status_label.setText(f"{len(camera_folders)}개의 카메라를 찾았습니다")
        else:
            self.status_label.setText("camera 폴더에 base.yaml을 가진 폴더가 없습니다")

        self.camera_combo.blockSignals(False)

        # Auto-select first camera
        if camera_folders and self.camera_combo.count() > 0:
            self.camera_combo.setCurrentIndex(0)
            self._on_camera_selected(0)

    def _on_camera_selected(self, index):
        """Load camera folder and base.yaml"""
        if index < 0:
            return

        camera_folder = self.camera_combo.currentData()
        if not camera_folder:
            return

        try:
            camera_folder_path = Path(camera_folder)
            camera_name = camera_folder_path.name

            # Load base.yaml
            base_yaml_path = camera_folder_path / "base.yaml"
            with open(base_yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # Update camera params
            self.camera_params = CameraParams(**data)
            self.current_camera_yaml = str(base_yaml_path)

            # Load map data
            shp_path = Path("shp") / camera_name

            if not shp_path.exists():
                self.status_label.setText(f"SHP 폴더를 찾을 수 없습니다: {shp_path}")
                return

            self.projector = MapProjector(shp_root=str(shp_path))
            self.projector.load_map_data(
                x=self.camera_params.x,
                y=self.camera_params.y,
                radius=500
            )

            # Populate controls
            self._populate_controls_from_params()

            # Load images
            self._populate_image_list(camera_name)

            self.status_label.setText(f"카메라 '{camera_name}' (base.yaml) 로드 완료")

        except Exception as e:
            self.status_label.setText(f"카메라 로드 오류: {e}")
            import traceback
            traceback.print_exc()

    def _populate_image_list(self, camera_name):
        """Load images from image/{camera_name}/ folder"""
        self.image_list.clear()

        image_dir = Path("image") / camera_name
        if not image_dir.exists():
            self.status_label.setText(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")
            return

        # Find image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(image_dir.glob(ext))

        if image_files:
            for img_file in sorted(image_files):
                self.image_list.addItem(img_file.name)
            self.status_label.setText(f"{len(image_files)}개의 이미지를 찾았습니다")
        else:
            self.status_label.setText(f"이미지 폴더에 이미지가 없습니다: {image_dir}")

    def _on_image_selected(self, current, previous):
        """Load and display selected image with map overlay"""
        if not current:
            return

        # Get camera folder and image info
        camera_folder = Path(self.current_camera_yaml).parent
        camera_name = camera_folder.name
        image_filename = current.text()
        image_name_no_ext = Path(image_filename).stem

        image_path = Path("image") / camera_name / image_filename
        if not image_path.exists():
            return

        self.current_image_path = str(image_path)

        # Try to find the best matching yaml file
        yaml_to_use = None
        yaml_source_msg = ""

        # 1. Try exact match: {image_name}.yaml
        image_yaml = camera_folder / f"{image_name_no_ext}.yaml"
        if image_yaml.exists():
            yaml_to_use = image_yaml
            yaml_source_msg = f"이미지 '{image_filename}' 전용 설정 로드"
        else:
            # 2. Try to find closest smaller numbered yaml
            # Extract number from image filename
            import re
            match = re.search(r'(\d+)', image_name_no_ext)
            if match:
                current_number = int(match.group(1))

                # Find all numbered yaml files in camera folder
                candidate_yamls = []
                for yaml_file in camera_folder.glob("*.yaml"):
                    if yaml_file.name == "base.yaml":
                        continue
                    yaml_match = re.search(r'(\d+)', yaml_file.stem)
                    if yaml_match:
                        yaml_number = int(yaml_match.group(1))
                        if yaml_number < current_number:
                            candidate_yamls.append((yaml_number, yaml_file))

                # Use the closest one (largest number less than current)
                if candidate_yamls:
                    candidate_yamls.sort(reverse=True)
                    yaml_to_use = candidate_yamls[0][1]
                    yaml_source_msg = f"이미지 '{image_filename}' ({yaml_to_use.name} 사용)"

            # 3. Fallback to base.yaml
            if yaml_to_use is None:
                yaml_to_use = camera_folder / "base.yaml"
                yaml_source_msg = f"이미지 '{image_filename}' (base.yaml 사용)"

        # Load the selected yaml
        try:
            with open(yaml_to_use, 'r') as f:
                data = yaml.safe_load(f)
            self.camera_params = CameraParams(**data)
            self.current_camera_yaml = str(yaml_to_use)
            self._populate_controls_from_params()
            self.status_label.setText(yaml_source_msg)
        except Exception as e:
            self.status_label.setText(f"YAML 로드 오류: {e}")
            import traceback
            traceback.print_exc()

        self._draw_image_with_overlay()

    def _populate_controls_from_params(self):
        """Populate UI controls from camera_params"""
        # Block all signals
        all_controls = list(self.position_controls.values()) + list(self.orientation_controls.values()) + list(self.intrinsic_controls.values())
        for ctrl in all_controls:
            ctrl['spinbox'].blockSignals(True)
            ctrl['slider'].blockSignals(True)

        # Set position values
        for name, val in [('X', self.camera_params.x),
                         ('Y', self.camera_params.y),
                         ('Z', self.camera_params.z)]:
            ctrl = self.position_controls[name]
            ctrl['base_value'] = val
            ctrl['spinbox'].setValue(val)
            ctrl['slider'].setValue(0)

        # Set orientation values
        slider_scale = 1000
        for name, val in [('Yaw', self.camera_params.yaw),
                         ('Pitch', self.camera_params.pitch),
                         ('Roll', self.camera_params.roll)]:
            ctrl = self.orientation_controls[name]
            ctrl['spinbox'].setValue(val)
            ctrl['slider'].setValue(int(val * slider_scale))

        # Set intrinsic values
        slider_scale = 100
        for name, val in [('fx', self.camera_params.fx),
                         ('fy', self.camera_params.fy)]:
            ctrl = self.intrinsic_controls[name]
            ctrl['spinbox'].setValue(val)
            ctrl['slider'].setValue(int(val * slider_scale))

        # Unblock signals
        for ctrl in all_controls:
            ctrl['spinbox'].blockSignals(False)
            ctrl['slider'].blockSignals(False)

        # Update projection
        self._update_projection()

    def _on_map_visibility_changed(self, checked):
        """Handle map visibility checkbox toggle"""
        self.show_map_lines = checked
        self._draw_image_with_overlay()

    def _update_projection(self):
        """Recalculate projection with current parameters"""
        if not self.projector:
            return

        # Update camera params from controls
        self.camera_params.x = self.position_controls['X']['spinbox'].value()
        self.camera_params.y = self.position_controls['Y']['spinbox'].value()
        self.camera_params.z = self.position_controls['Z']['spinbox'].value()
        self.camera_params.yaw = self.orientation_controls['Yaw']['spinbox'].value()
        self.camera_params.pitch = self.orientation_controls['Pitch']['spinbox'].value()
        self.camera_params.roll = self.orientation_controls['Roll']['spinbox'].value()
        self.camera_params.fx = self.intrinsic_controls['fx']['spinbox'].value()
        self.camera_params.fy = self.intrinsic_controls['fy']['spinbox'].value()

        # Recalculate projection
        self.projected_map_data = self.projector.projection(self.camera_params).projected_layers

        # Redraw image
        self._draw_image_with_overlay()

    def _draw_image_with_overlay(self):
        """Draw image with map projection overlay"""
        if not self.current_image_path:
            return

        # Load image
        image = cv2.imread(self.current_image_path)
        if image is None:
            return

        # Draw projection overlay (only if checkbox is checked)
        if self.show_map_lines and self.projected_map_data:
            # Draw b2 (lines) in cyan
            if self.projected_map_data.get('b2'):
                cv2.polylines(image, self.projected_map_data['b2'],
                            isClosed=False, color=(255, 255, 0), thickness=3)

        # Convert to Qt and display
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _save_camera_yaml(self):
        """Save current camera parameters to image-specific YAML"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        try:
            # Get camera folder and image name
            camera_folder = Path(self.current_camera_yaml).parent
            image_filename = Path(self.current_image_path).name
            image_name_no_ext = Path(image_filename).stem

            # Save path: camera/{camera_name}/{image_name}.yaml
            save_path = camera_folder / f"{image_name_no_ext}.yaml"

            # Update camera params from controls
            self.camera_params.x = self.position_controls['X']['spinbox'].value()
            self.camera_params.y = self.position_controls['Y']['spinbox'].value()
            self.camera_params.z = self.position_controls['Z']['spinbox'].value()
            self.camera_params.yaw = self.orientation_controls['Yaw']['spinbox'].value()
            self.camera_params.pitch = self.orientation_controls['Pitch']['spinbox'].value()
            self.camera_params.roll = self.orientation_controls['Roll']['spinbox'].value()
            self.camera_params.fx = self.intrinsic_controls['fx']['spinbox'].value()
            self.camera_params.fy = self.intrinsic_controls['fy']['spinbox'].value()

            # Convert to dict
            data = {
                'x': self.camera_params.x,
                'y': self.camera_params.y,
                'z': self.camera_params.z,
                'yaw': self.camera_params.yaw,
                'pitch': self.camera_params.pitch,
                'roll': self.camera_params.roll,
                'fx': self.camera_params.fx,
                'fy': self.camera_params.fy,
                'cx': self.camera_params.cx,
                'cy': self.camera_params.cy,
                'k1': self.camera_params.k1,
                'k2': self.camera_params.k2,
                'p1': self.camera_params.p1,
                'p2': self.camera_params.p2,
                'k3': self.camera_params.k3,
                'resolution_width': self.camera_params.resolution_width,
                'resolution_height': self.camera_params.resolution_height,
            }

            # Save to YAML
            with open(save_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            self.current_camera_yaml = str(save_path)
            self.status_label.setText(f"저장 완료: {save_path.name}")

        except Exception as e:
            self.status_label.setText(f"저장 오류: {e}")

    def _reset_to_yaml(self):
        """Reset controls to values from YAML file (image-specific or base)"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        try:
            # Get camera folder and image name
            camera_folder = Path(self.current_camera_yaml).parent
            image_filename = Path(self.current_image_path).name
            image_name_no_ext = Path(image_filename).stem

            # Try to load image-specific yaml, fallback to base.yaml
            image_yaml = camera_folder / f"{image_name_no_ext}.yaml"
            if image_yaml.exists():
                yaml_path = image_yaml
                msg = f"이미지 전용 설정({image_yaml.name})에서 리셋"
            else:
                yaml_path = camera_folder / "base.yaml"
                msg = "base.yaml에서 리셋"

            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            self.camera_params = CameraParams(**data)
            self.current_camera_yaml = str(yaml_path)
            self._populate_controls_from_params()
            self.status_label.setText(msg)

        except Exception as e:
            self.status_label.setText(f"리셋 오류: {e}")

    def _create_vision_group(self):
        """Create a group box for computer vision tasks"""
        group = QGroupBox("Vision Tasks")
        layout = QHBoxLayout()

        self.btn_compare_features = QPushButton("특징점 매칭\n(SIFT)")
        self.btn_compare_features.clicked.connect(self._on_compare_features_clicked)

        self.btn_compare_features_orb = QPushButton("특징점 매칭\n(ORB)")
        self.btn_compare_features_orb.clicked.connect(self._on_compare_features_orb_clicked)

        self.btn_detect_vp = QPushButton("소실점 검출\n(LSD + RANSAC)")
        self.btn_detect_vp.clicked.connect(self._on_detect_vanishing_point_clicked)

        layout.addWidget(self.btn_compare_features)
        layout.addWidget(self.btn_compare_features_orb)
        layout.addWidget(self.btn_detect_vp)

        # Future buttons can be added here
        layout.addStretch()

        group.setLayout(layout)
        return group

    def _on_detect_vanishing_point_clicked(self):
        """Open vanishing point detection dialog"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        # Import here to avoid circular dependency
        from app.ui.vanishing_point_dialog import VanishingPointDialog

        try:
            dialog = VanishingPointDialog(self.current_image_path, self)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"소실점 검출 오류: {e}")
            import traceback
            traceback.print_exc()


    def _on_compare_features_clicked(self):
        """Open file dialog to select 2 images and match them"""
        # Import here to avoid circular dependency
        from app.ui.matcher_dialog import MatcherDialog

        # Default dir
        if self.current_image_path:
            init_dir = str(Path(self.current_image_path).parent)
        else:
            init_dir = os.getcwd()

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "비교할 이미지 2개 선택",
            init_dir,
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not files:
            return

        if len(files) != 2:
            self.status_label.setText(f"이미지를 정확히 2개 선택해야 합니다. (선택된 개수: {len(files)})")
            return

        try:
            dialog = MatcherDialog(files[0], files[1], self)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"매칭 오류: {e}")
            import traceback
            traceback.print_exc()

    def _on_compare_features_orb_clicked(self):
        """Open file dialog to select 2 images and match them using ORB"""
        # Import here to avoid circular dependency
        from app.ui.matcher_dialog_orb import MatcherDialogORB

        # Default dir
        if self.current_image_path:
            init_dir = str(Path(self.current_image_path).parent)
        else:
            init_dir = os.getcwd()

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "비교할 이미지 2개 선택 (ORB)",
            init_dir,
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not files:
            return

        if len(files) != 2:
            self.status_label.setText(f"이미지를 정확히 2개 선택해야 합니다. (선택된 개수: {len(files)})")
            return

        try:
            dialog = MatcherDialogORB(files[0], files[1], self)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"ORB 매칭 오류: {e}")
            import traceback
            traceback.print_exc()
