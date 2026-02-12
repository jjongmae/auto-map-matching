"""
맵 매칭 애플리케이션을 위한 메인 윈도우
"""
import os
import time
import yaml
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton,
    QSlider, QDoubleSpinBox, QGroupBox, QHBoxLayout, QVBoxLayout,
    QGridLayout, QFileDialog, QListWidget, QCheckBox,
    QDialog, QListWidgetItem, QDialogButtonBox, QLineEdit
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage
import cv2
import numpy as np

from u1gis_geovision import MapProjector, CameraParams


class CameraSelectionDialog(QDialog):
    """카메라 선택을 위한 다이얼로그 (ComboBox 대체)"""
    def __init__(self, cameras, parent=None):
        super().__init__(parent)
        self.setWindowTitle("카메라 선택")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        # 검색 필터
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("카메라 이름 검색...")
        self.search_edit.textChanged.connect(self._filter_list)
        layout.addWidget(self.search_edit)

        # 리스트 위젯
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)

        # 버튼 박스
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.cameras = cameras
        self.selected_camera_folder = None
        self.selected_camera_name = None

        self._populate_list()

        # 더블 클릭 시 선택 처리
        self.list_widget.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _populate_list(self):
        self.list_widget.clear()
        for cam in self.cameras:
            item_text = cam['name']
            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, cam)  # 전체 데이터 저장
            self.list_widget.addItem(item)

    def _filter_list(self, text):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            # 대소문자 구분 없이 검색
            if text.lower() in item.text().lower():
                item.setHidden(False)
            else:
                item.setHidden(True)

    def _on_item_double_clicked(self, item):
        self.accept()

    def accept(self):
        # 선택된 아이템 확인
        selected_items = self.list_widget.selectedItems()
        if selected_items:
            cam_data = selected_items[0].data(Qt.UserRole)
            self.selected_camera_folder = cam_data['folder']
            self.selected_camera_name = cam_data['name']
        super().accept()


class MapMatcherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Map Matcher")
        self.setGeometry(100, 100, 1600, 900)

        # --- 데이터 속성 ---
        self.position_controls = {}
        self.orientation_controls = {}
        self.intrinsic_controls = {}
        self.current_image_path = None
        self.current_camera_yaml = None
        self.show_map_lines = True  # 기본값: 지도 선 표시

        # VAID GIS
        self.projector = None
        self.projected_map_data = None
        self.camera_params = CameraParams()

        # 카메라 목록 데이터
        self.camera_data_list = []
        

        # --- UI 설정 ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(10)

        # 메인 콘텐츠: 이미지 (왼쪽) + 제어 (오른쪽)
        main_content = QHBoxLayout()
        root_layout.addLayout(main_content)

        # 왼쪽: 이미지 표시
        image_panel = QVBoxLayout()
        self.image_label = QLabel("이미지를 선택하세요")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black; color: grey; font-size: 20px;")
        self.image_label.setFixedSize(1280, 720)
        image_panel.addWidget(self.image_label)
        image_panel.addWidget(self._create_vision_group())
        main_content.addLayout(image_panel, 7)

        # 오른쪽: 제어
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

        # 상태 표시줄
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

        # 초기 카메라 목록 채우기
        self._populate_camera_list()

    def _create_camera_selection_group(self):
        group = QGroupBox("카메라 선택 (YAML)")
        layout = QHBoxLayout()

        self.camera_name_edit = QLineEdit()
        self.camera_name_edit.setReadOnly(True)
        self.camera_name_edit.setPlaceholderText("카메라를 선택하세요")

        select_button = QPushButton("선택...")
        select_button.clicked.connect(self._open_camera_selection_dialog)

        layout.addWidget(self.camera_name_edit)
        layout.addWidget(select_button)
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
            
            # Yaw와 Roll은 360도 회전이므로 값이 순환되도록 설정 (180 -> -180)
            if name in ["Yaw", "Roll"]:
                spinbox.setWrapping(True)
                
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

            spinbox.setRange(100, 100000)
            spinbox.setDecimals(2)
            spinbox.setSingleStep(1)

            slider_scale = 100
            slider.setRange(100 * slider_scale, 100000 * slider_scale)

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

        # 지도 선 표시/숨기기를 위한 체크박스
        self.show_map_checkbox = QCheckBox("지도 선 표시")
        self.show_map_checkbox.setChecked(True)  # 기본값: 선택됨
        self.show_map_checkbox.toggled.connect(self._on_map_visibility_changed)
        layout.addWidget(self.show_map_checkbox)

        # 버튼
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
        """base.yaml을 포함하는 하위 폴더에 대해 camera 폴더 스캔"""
        self.camera_data_list = []

        camera_dir = Path("camera")
        if not camera_dir.exists():
            self.status_label.setText("camera 폴더가 없습니다")
            return

        # base.yaml이 있는 폴더 찾기
        camera_folders = []
        for folder in camera_dir.iterdir():
            if folder.is_dir():
                base_yaml = folder / "base.yaml"
                if base_yaml.exists():
                    camera_folders.append(folder)

        if camera_folders:
            for folder in sorted(camera_folders):
                self.camera_data_list.append({
                    'name': folder.name,
                    'folder': str(folder)
                })
            self.status_label.setText(f"{len(camera_folders)}개의 카메라를 찾았습니다")
        else:
            self.status_label.setText("camera 폴더에 base.yaml을 가진 폴더가 없습니다")

        # 첫 번째 카메라 자동 선택
        if self.camera_data_list:
            first_camera = self.camera_data_list[0]
            self.camera_name_edit.setText(first_camera['name'])
            self._on_camera_selected(first_camera['folder'])

    def _open_camera_selection_dialog(self):
        """카메라 선택 다이얼로그 열기"""
        # 항상 최신 목록을 가져옴
        self._populate_camera_list()

        dialog = CameraSelectionDialog(self.camera_data_list, self)
        if dialog.exec() == QDialog.Accepted:
            if dialog.selected_camera_folder:
                self.camera_name_edit.setText(dialog.selected_camera_name)
                self._on_camera_selected(dialog.selected_camera_folder)

    def _on_camera_selected(self, camera_folder):
        """카메라 폴더 및 base.yaml 로드"""
        if not camera_folder:
            return

        try:
            camera_folder_path = Path(camera_folder)
            camera_name = camera_folder_path.name

            # base.yaml 로드
            base_yaml_path = camera_folder_path / "base.yaml"
            with open(base_yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # 카메라 파라미터 업데이트
            self.camera_params = CameraParams(**data)
            self.current_camera_yaml = str(base_yaml_path)

            # 지도 데이터 로드
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

            # 제어 항목 채우기
            self._populate_controls_from_params()

            # 이미지 로드
            self._populate_image_list(camera_name)

            self.status_label.setText(f"카메라 '{camera_name}' (base.yaml) 로드 완료")

        except Exception as e:
            self.status_label.setText(f"카메라 로드 오류: {e}")
            import traceback
            traceback.print_exc()

    def _populate_image_list(self, camera_name):
        """image/{camera_name}/ 폴더에서 이미지 로드"""
        self.image_list.clear()

        image_dir = Path("image") / camera_name
        if not image_dir.exists():
            self.status_label.setText(f"이미지 폴더를 찾을 수 없습니다: {image_dir}")
            return

        # 이미지 파일 찾기
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
        """지도 오버레이와 함께 선택한 이미지 로드 및 표시"""
        if not current:
            return

        # 카메라 폴더 및 이미지 정보 가져오기
        camera_folder = Path(self.current_camera_yaml).parent
        camera_name = camera_folder.name
        image_filename = current.text()
        image_name_no_ext = Path(image_filename).stem

        image_path = Path("image") / camera_name / image_filename
        if not image_path.exists():
            return

        self.current_image_path = str(image_path)
        

        # 가장 잘 일치하는 yaml 파일 찾기 시도
        yaml_to_use = None
        yaml_source_msg = ""

        # 1. 정확한 일치 시도: {image_name}.yaml
        image_yaml = camera_folder / f"{image_name_no_ext}.yaml"
        if image_yaml.exists():
            yaml_to_use = image_yaml
            yaml_source_msg = f"이미지 '{image_filename}' 전용 설정 로드"
        else:
            # 2. 가장 가까운 더 작은 번호의 yaml 찾기 시도
            # 이미지 파일 이름에서 번호 추출
            import re
            match = re.search(r'(\d+)', image_name_no_ext)
            if match:
                current_number = int(match.group(1))

                # 카메라 폴더의 모든 번호가 매겨진 yaml 파일 찾기
                candidate_yamls = []
                for yaml_file in camera_folder.glob("*.yaml"):
                    if yaml_file.name == "base.yaml":
                        continue
                    yaml_match = re.search(r'(\d+)', yaml_file.stem)
                    if yaml_match:
                        yaml_number = int(yaml_match.group(1))
                        if yaml_number < current_number:
                            candidate_yamls.append((yaml_number, yaml_file))

                # 가장 가까운 것 사용 (현재보다 작은 가장 큰 번호)
                if candidate_yamls:
                    candidate_yamls.sort(reverse=True)
                    yaml_to_use = candidate_yamls[0][1]
                    yaml_source_msg = f"이미지 '{image_filename}' ({yaml_to_use.name} 사용)"

            # 3. base.yaml로 대체
            if yaml_to_use is None:
                yaml_to_use = camera_folder / "base.yaml"
                yaml_source_msg = f"이미지 '{image_filename}' (base.yaml 사용)"

        # 선택된 yaml 로드
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
        """camera_params에서 UI 제어 항목 채우기"""
        # 모든 신호 차단
        all_controls = list(self.position_controls.values()) + list(self.orientation_controls.values()) + list(self.intrinsic_controls.values())
        for ctrl in all_controls:
            ctrl['spinbox'].blockSignals(True)
            ctrl['slider'].blockSignals(True)

        # 위치 값 설정
        for name, val in [('X', self.camera_params.x),
                         ('Y', self.camera_params.y),
                         ('Z', self.camera_params.z)]:
            ctrl = self.position_controls[name]
            ctrl['base_value'] = val
            ctrl['spinbox'].setValue(val)
            ctrl['slider'].setValue(0)

        # 방향 값 설정
        slider_scale = 1000
        for name, val in [('Yaw', self.camera_params.yaw),
                         ('Pitch', self.camera_params.pitch),
                         ('Roll', self.camera_params.roll)]:
            ctrl = self.orientation_controls[name]
            ctrl['spinbox'].setValue(val)
            ctrl['slider'].setValue(int(val * slider_scale))

        # 내부 파라미터 값 설정
        slider_scale = 100
        for name, val in [('fx', self.camera_params.fx),
                         ('fy', self.camera_params.fy)]:
            ctrl = self.intrinsic_controls[name]
            ctrl['spinbox'].setValue(val)
            ctrl['slider'].setValue(int(val * slider_scale))

        # 신호 차단 해제
        for ctrl in all_controls:
            ctrl['spinbox'].blockSignals(False)
            ctrl['slider'].blockSignals(False)

        # 투영 업데이트
        self._update_projection()

    def _on_map_visibility_changed(self, checked):
        """지도 가시성 체크박스 토글 처리"""
        self.show_map_lines = checked
        self._draw_image_with_overlay()

    def _update_projection(self):
        """현재 파라미터로 투영 재계산"""
        if not self.projector:
            return

        # 제어 항목에서 카메라 파라미터 업데이트
        self.camera_params.x = self.position_controls['X']['spinbox'].value()
        self.camera_params.y = self.position_controls['Y']['spinbox'].value()
        self.camera_params.z = self.position_controls['Z']['spinbox'].value()
        self.camera_params.yaw = self.orientation_controls['Yaw']['spinbox'].value()
        self.camera_params.pitch = self.orientation_controls['Pitch']['spinbox'].value()
        self.camera_params.roll = self.orientation_controls['Roll']['spinbox'].value()
        self.camera_params.fx = self.intrinsic_controls['fx']['spinbox'].value()
        self.camera_params.fy = self.intrinsic_controls['fy']['spinbox'].value()

        # 투영 재계산
        self.projected_map_data = self.projector.projection(self.camera_params).projected_layers

        # 이미지 다시 그리기
        self._draw_image_with_overlay()

    def _draw_image_with_overlay(self):
        """지도 투영 오버레이로 이미지 그리기"""
        if not self.current_image_path:
            return

        # 이미지 로드
        image = cv2.imread(self.current_image_path)
        if image is None:
            return

        # 투영 오버레이 그리기 (체크박스가 선택된 경우에만)
        if self.show_map_lines and self.projected_map_data:
            # b2 (선)를 파란색으로 그리기 (선 + 점)
            if self.projected_map_data.get('b2'):
                for line in self.projected_map_data['b2']:
                    if len(line) > 0:
                        # 선 그리기
                        line_arr = np.array(line, dtype=np.int32)
                        cv2.polylines(image, [line_arr], isClosed=False, color=(255, 0, 0), thickness=2)

        # Qt로 변환 및 표시
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        scaled = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)

    def _save_camera_yaml(self):
        """현재 카메라 파라미터를 이미지별 YAML에 저장"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        try:
            # 카메라 폴더 및 이미지 이름 가져오기
            camera_folder = Path(self.current_camera_yaml).parent
            image_filename = Path(self.current_image_path).name
            image_name_no_ext = Path(image_filename).stem

            # 저장 경로: camera/{camera_name}/{image_name}.yaml
            save_path = camera_folder / f"{image_name_no_ext}.yaml"

            # 제어 항목에서 카메라 파라미터 업데이트
            self.camera_params.x = self.position_controls['X']['spinbox'].value()
            self.camera_params.y = self.position_controls['Y']['spinbox'].value()
            self.camera_params.z = self.position_controls['Z']['spinbox'].value()
            self.camera_params.yaw = self.orientation_controls['Yaw']['spinbox'].value()
            self.camera_params.pitch = self.orientation_controls['Pitch']['spinbox'].value()
            self.camera_params.roll = self.orientation_controls['Roll']['spinbox'].value()
            self.camera_params.fx = self.intrinsic_controls['fx']['spinbox'].value()
            self.camera_params.fy = self.intrinsic_controls['fy']['spinbox'].value()

            # 딕셔너리로 변환
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

            # YAML로 저장
            with open(save_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)

            self.current_camera_yaml = str(save_path)
            self.status_label.setText(f"저장 완료: {save_path.name}")

        except Exception as e:
            self.status_label.setText(f"저장 오류: {e}")

    def _reset_to_yaml(self):
        """YAML 파일(이미지별 또는 base)의 값으로 제어 항목 초기화"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        try:
            # 카메라 폴더 및 이미지 이름 가져오기
            camera_folder = Path(self.current_camera_yaml).parent
            image_filename = Path(self.current_image_path).name
            image_name_no_ext = Path(image_filename).stem

            # 이미지별 yaml 로드 시도, base.yaml로 대체
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
        """컴퓨터 비전 작업을 위한 그룹 박스 생성"""
        group = QGroupBox("Vision Tasks")
        layout = QHBoxLayout()

        self.btn_compare_features = QPushButton("특징점 매칭\n(SIFT)")
        self.btn_compare_features.clicked.connect(self._on_compare_features_clicked)

        self.btn_compare_features_orb = QPushButton("특징점 매칭\n(ORB)")
        self.btn_compare_features_orb.clicked.connect(self._on_compare_features_orb_clicked)



        self.btn_lane_annotation = QPushButton("차선 라벨링\n(수동)")
        self.btn_lane_annotation.clicked.connect(self._on_lane_annotation_clicked)

        self.btn_lane_detection_sam3 = QPushButton("차선 검출\n(SAM3)")
        self.btn_lane_detection_sam3.clicked.connect(self._on_lane_detection_sam3_clicked)

        self.btn_lane_fit_powell = QPushButton("차선 피팅\n(Powell)")
        self.btn_lane_fit_powell.clicked.connect(lambda: self._on_lane_fit_clicked('powell'))

        self.btn_lane_fit_nm = QPushButton("차선 피팅\n(NM)")
        self.btn_lane_fit_nm.clicked.connect(lambda: self._on_lane_fit_clicked('nelder_mead'))

        self.btn_lane_fit_lm = QPushButton("차선 피팅\n(LM)")
        self.btn_lane_fit_lm.clicked.connect(lambda: self._on_lane_fit_clicked('lm'))

        self.btn_lane_fit_de = QPushButton("차선 피팅\n(DE)")
        self.btn_lane_fit_de.clicked.connect(lambda: self._on_lane_fit_clicked('differential_evolution'))

        self.btn_lane_fit_de_parallel = QPushButton("차선 피팅\n(DE 병렬)")
        self.btn_lane_fit_de_parallel.clicked.connect(lambda: self._on_lane_fit_clicked('differential_evolution_parallel'))

        layout.addWidget(self.btn_compare_features)
        layout.addWidget(self.btn_compare_features_orb)

        layout.addWidget(self.btn_lane_annotation)
        layout.addWidget(self.btn_lane_detection_sam3)
        layout.addWidget(self.btn_lane_fit_powell)
        layout.addWidget(self.btn_lane_fit_nm)
        layout.addWidget(self.btn_lane_fit_lm)
        layout.addWidget(self.btn_lane_fit_de)
        layout.addWidget(self.btn_lane_fit_de_parallel)

        layout.addStretch()

        group.setLayout(layout)
        return group




    def _on_compare_features_clicked(self):
        """파일 대화상자를 열어 2개의 이미지를 선택하고 매칭합니다."""
        # 순환 종속성을 피하기 위해 여기서 임포트
        from app.ui.matcher_dialog import MatcherDialog

        # 기본 디렉토리
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
            dialog = MatcherDialog(files[0], files[1], self, projector=self.projector)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"매칭 오류: {e}")
            import traceback
            traceback.print_exc()

    def _on_compare_features_orb_clicked(self):
        """파일 대화상자를 열어 2개의 이미지를 선택하고 ORB를 사용하여 매칭합니다."""
        # 순환 종속성을 피하기 위해 여기서 임포트
        from app.ui.matcher_dialog_orb import MatcherDialogORB

        # 기본 디렉토리
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
            dialog = MatcherDialogORB(files[0], files[1], self, projector=self.projector)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"ORB 매칭 오류: {e}")
            import traceback
            traceback.print_exc()

    def _on_lane_annotation_clicked(self):
        """차선 정답지 생성 대화상자 열기"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        from app.ui.lane_labeling_dialog import LaneLabelingDialog

        try:
            dialog = LaneLabelingDialog(self.current_image_path, self)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"차선 라벨링 오류: {e}")
            import traceback
            traceback.print_exc()

    def _on_lane_detection_sam3_clicked(self):
        """SAM3를 사용한 차선 검출 대화상자 열기"""
        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        from app.ui.lane_detection_sam3_dialog import LaneDetectionSAM3Dialog

        try:
            dialog = LaneDetectionSAM3Dialog(self.current_image_path, self)
            dialog.exec()
        except Exception as e:
            self.status_label.setText(f"SAM3 차선 검출 오류: {e}")
            import traceback
            traceback.print_exc()

    def _normalize_angle(self, angle):
        """각도를 -180 ~ 180 범위로 정규화 (yaw, roll용)"""
        # 각도를 -180 ~ 180 범위로 변환
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle

    def _normalize_pitch(self, pitch):
        """pitch를 -90 ~ 90 범위로 정규화"""
        # pitch는 -90 ~ 90 범위로 제한
        # 범위를 벗어나면 클램핑
        if pitch > 90:
            return 90
        elif pitch < -90:
            return -90
        return pitch

    def _on_lane_fit_clicked(self, algorithm='powell'):
        """라벨링된 차선에 맞춰 카메라 파라미터 자동 피팅"""
        import json

        algorithm_names = {
            'powell': 'Powell',
            'nelder_mead': 'NM',
            'lm': 'LM',
            'differential_evolution': 'DE',
            'differential_evolution_parallel': 'DE (병렬)'
        }

        if not self.current_image_path:
            self.status_label.setText("이미지를 먼저 선택하세요")
            return

        if not self.projector:
            self.status_label.setText("지도 데이터가 로드되지 않았습니다")
            return

        # 차선 라벨링 데이터 경로 계산
        img_path = Path(self.current_image_path)
        parts = list(img_path.parts)

        if 'image' in parts:
            idx = parts.index('image')
            parts[idx] = 'lane_gt'
            json_path = Path(*parts).with_suffix('.json')
        else:
            self.status_label.setText("이미지 경로에서 lane_gt 폴더를 찾을 수 없습니다")
            return

        # 라벨링 데이터 로드
        if not json_path.exists():
            self.status_label.setText(f"차선 라벨링 파일이 없습니다: {json_path.name}")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"[피팅] JSON 로드: {json_path}")
            print(f"[피팅] JSON keys: {list(data.keys())}")
        except Exception as e:
            self.status_label.setText(f"라벨링 파일 로드 오류: {e}")
            return

        # 두 가지 JSON 구조 지원:
        # 1. 새 구조: {"lanes": [{"points": [...]}]}
        # 2. 기존 구조: {"points": [...]}
        all_points = []

        if 'lanes' in data:
            # 새 구조
            lanes = data.get('lanes', [])
            for lane in lanes:
                all_points.extend(lane.get('points', []))
            print(f"[피팅] 새 구조 (lanes): {len(lanes)}개 그룹, 총 {len(all_points)}개 점")
        elif 'points' in data:
            # 기존 구조
            all_points = data.get('points', [])
            print(f"[피팅] 기존 구조 (points): {len(all_points)}개 점")

        if not all_points:
            self.status_label.setText("라벨링된 차선이 없습니다")
            return

        if len(all_points) < 2:
            self.status_label.setText("라벨링 점이 부족합니다 (최소 2개 필요)")
            return


        
        # --- 디버깅 로그 (필요시 주석 해제) ---
        # print("\n" + "="*50)
        # print("[라인 피팅 시작 로그]")
        # 
        # # 1. 라벨링 데이터 (그룹 점)
        # print(f"\n1. 라벨링 데이터 (총 {len(lanes)}개 그룹)")
        # for i, lane in enumerate(lanes):
        #     pts = lane.get('points', [])
        #     print(f"   - 그룹 {i} (점 {len(pts)}개): {pts}")
        # 
        # # 2. 투영된 지도 데이터 (그룹 점)
        # print("\n2. 투영된 지도 데이터 (초기 상태 - 화면 내 유효 점만 표시)")
        # 
        # # 로그용 초기 투영 데이터 계산
        # initial_b2_lines = []
        # try:
        #     proj_result = self.projector.projection(self.camera_params, target_layers=['b2'], as_float=True, include_cache=False)
        #     initial_b2_lines = proj_result.get('b2', []) if isinstance(proj_result, dict) else getattr(proj_result, 'projected_layers', {}).get('b2', [])
        # except Exception:
        #     pass
        #     
        # # 해상도 기준 설정 (1920x1080)
        # res_w = getattr(self.camera_params, 'resolution_width', 0) or 1920
        # res_h = getattr(self.camera_params, 'resolution_height', 0) or 1080
        # 
        # print(f"   - 해상도 기준: {res_w} x {res_h} (마진 0px)")
        # print(f"   - 원본 투영 라인 그룹 수: {len(initial_b2_lines)}")
        # 
        # valid_total_count = 0
        # 
        # for i, line in enumerate(initial_b2_lines):
        #     if len(line) == 0:
        #         continue
        #         
        #     pts_arr = np.array(line, dtype=np.float64)
        #     
        #     # 필터링 로직
        #     mask = (
        #         (pts_arr[:, 0] >= 0) & (pts_arr[:, 0] <= res_w) &
        #         (pts_arr[:, 1] >= 0) & (pts_arr[:, 1] <= res_h)
        #     )
        #     valid_pts = pts_arr[mask]
        #     
        #     valid_total_count += len(valid_pts)
        #     
        #     if len(valid_pts) > 0:
        #         print(f"   - 그룹 {i} (유효 {len(valid_pts)}/{len(line)}개): {valid_pts.tolist()}")
        #     else:
        #         print(f"   - 그룹 {i} (유효 0/{len(line)}개): [모두 화면 밖 - 제외됨]")
        # 
        # print(f"   => 총 최적화 사용 예정 점 개수: {valid_total_count}")
        # print("="*50 + "\n")
        # --- 로그 기록 종료 ---

        self.status_label.setText(f"차선 피팅 중... ({algorithm_names.get(algorithm, algorithm)})")

        # 알고리즘별 함수 선택
        from app.core.auto_fitter import fit_powell, fit_nelder_mead, fit_lm, fit_differential_evolution, fit_differential_evolution_parallel

        fit_functions = {
            'powell': fit_powell,
            'nelder_mead': fit_nelder_mead,
            'lm': fit_lm,
            'differential_evolution': fit_differential_evolution,
            'differential_evolution_parallel': fit_differential_evolution_parallel
        }

        fit_func = fit_functions.get(algorithm, fit_powell)

        start_time = time.time()
        try:
            result = fit_func(self.projector, self.camera_params, all_points)
            elapsed_time = time.time() - start_time

            if result is None:
                self.status_label.setText("차선 피팅 실패")
                return

            # 최적화된 파라미터로 UI 업데이트
            self.camera_params.yaw = self._normalize_angle(result['yaw'])
            self.camera_params.pitch = self._normalize_pitch(result['pitch'])
            self.camera_params.roll = self._normalize_angle(result['roll'])
            self.camera_params.fx = result['fx']
            self.camera_params.fy = result['fy']

            # 컨트롤 값 업데이트 (자동으로 _draw_image_with_overlay 호출됨)
            self._populate_controls_from_params()
            
            # 수행 시간 출력
            print(f"[{algorithm_names.get(algorithm, algorithm)}] 피팅 수행 시간: {elapsed_time:.4f}초")

            self.status_label.setText(
                f"[{algorithm_names.get(algorithm, algorithm)}] 완료 ({elapsed_time:.3f}초) - "
                f"yaw: {self.camera_params.yaw:.2f}, pitch: {self.camera_params.pitch:.2f}, "
                f"roll: {self.camera_params.roll:.2f}, fx: {result['fx']:.1f}, fy: {result['fy']:.1f}"
            )

        except Exception as e:
            self.status_label.setText(f"차선 피팅 오류: {e}")
            import traceback
            traceback.print_exc()
