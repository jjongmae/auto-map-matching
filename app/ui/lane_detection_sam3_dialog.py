"""
SAM3 차선 검출 결과를 표시하고 저장하는 다이얼로그
"""
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QScrollArea, QWidget,
    QProgressDialog, QComboBox, QSpinBox, QGroupBox, QLineEdit, QDoubleSpinBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter
import cv2
import numpy as np


class ZoomableImageWidget(QWidget):
    """확대/축소 지원 이미지 위젯"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._zoom = 1.0

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self._update_size()

    def setZoom(self, zoom):
        self._zoom = max(0.1, min(zoom, 10.0))
        self._update_size()

    def getZoom(self):
        return self._zoom

    def _update_size(self):
        if self._pixmap and not self._pixmap.isNull():
            orig_size = self._pixmap.size()
            new_w = int(orig_size.width() * self._zoom)
            new_h = int(orig_size.height() * self._zoom)
            self.setFixedSize(new_w, new_h)
        self.update()

    def paintEvent(self, event):
        if not self._pixmap or self._pixmap.isNull():
            super().paintEvent(event)
            return

        painter = QPainter(self)
        scaled = self._pixmap.scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        painter.drawPixmap(0, 0, scaled)


class DetectionWorker(QThread):
    """SAM3 검출을 백그라운드에서 수행하는 워커"""
    finished = Signal(list)  # 검출된 차선 리스트
    error = Signal(str)      # 에러 메시지

    def __init__(self, image_path, text_prompts, conf=0.5, poly_degree=2):
        super().__init__()
        self.image_path = image_path
        self.text_prompts = text_prompts
        self.conf = conf
        self.poly_degree = poly_degree

    def run(self):
        try:
            from app.core.lane_detector_sam3 import LaneDetectorSAM3

            detector = LaneDetectorSAM3(
                model_path="models/sam3.pt",
                device="cuda",
                conf=self.conf
            )
            lanes = detector.detect_lanes(
                self.image_path,
                text_prompts=self.text_prompts,
                poly_degree=self.poly_degree
            )
            self.finished.emit(lanes)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class LaneDetectionSAM3Dialog(QDialog):
    """SAM3 차선 검출 다이얼로그"""

    # 차선별 색상
    LANE_COLORS = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (255, 128, 0), (128, 0, 255), (0, 255, 128), (255, 128, 128),
    ]

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("차선 검출 (SAM3)")
        self.resize(1400, 900)

        self.image_path = image_path
        self.original_image = None
        self.original_size = None
        self.detected_lanes = []
        self.worker = None

        self._setup_ui()
        self._load_image()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 상단 정보 및 설정
        top_layout = QHBoxLayout()

        self.info_label = QLabel("SAM3로 차선을 검출합니다. [검출 시작] 버튼을 클릭하세요.")
        self.info_label.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;"
        )
        top_layout.addWidget(self.info_label, stretch=1)

        layout.addLayout(top_layout)

        # 설정 그룹
        settings_group = QGroupBox("검출 설정")
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("프롬프트:"))
        self.prompt_edit = QLineEdit("lane line")
        self.prompt_edit.setMinimumWidth(200)
        settings_layout.addWidget(self.prompt_edit)

        settings_layout.addWidget(QLabel("신뢰도:"))
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.5)
        self.conf_spinbox.setDecimals(2)
        self.conf_spinbox.setMinimumWidth(80)
        settings_layout.addWidget(self.conf_spinbox)

        settings_layout.addWidget(QLabel("라인 보정:"))
        self.smoothing_combo = QComboBox()
        self.smoothing_combo.addItem("안함", 0)
        self.smoothing_combo.addItem("직선으로", 1)
        self.smoothing_combo.addItem("부드러운 곡선", 2)
        self.smoothing_combo.addItem("복잡한 곡선", 3)
        self.smoothing_combo.setCurrentIndex(2)  # 기본값: 부드러운 곡선
        self.smoothing_combo.setMinimumWidth(120)
        settings_layout.addWidget(self.smoothing_combo)

        settings_layout.addStretch()

        self.btn_detect = QPushButton("검출 시작")
        self.btn_detect.clicked.connect(self._start_detection)
        self.btn_detect.setStyleSheet("padding: 8px 16px;")
        settings_layout.addWidget(self.btn_detect)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # 이미지 표시 영역
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.image_widget = ZoomableImageWidget()
        self.scroll_area.setWidget(self.image_widget)

        layout.addWidget(self.scroll_area, stretch=1)

        # 하단 버튼
        button_layout = QHBoxLayout()

        # 줌 컨트롤
        btn_zoom_out = QPushButton("축소 (-)")
        btn_zoom_out.clicked.connect(lambda: self._adjust_zoom(-0.25))
        button_layout.addWidget(btn_zoom_out)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(self.zoom_label)

        btn_zoom_in = QPushButton("확대 (+)")
        btn_zoom_in.clicked.connect(lambda: self._adjust_zoom(0.25))
        button_layout.addWidget(btn_zoom_in)

        btn_zoom_fit = QPushButton("맞춤")
        btn_zoom_fit.clicked.connect(self._fit_to_window)
        button_layout.addWidget(btn_zoom_fit)

        button_layout.addStretch()

        # 결과 레이블
        self.result_label = QLabel("검출된 차선: 0개")
        button_layout.addWidget(self.result_label)

        button_layout.addStretch()

        # 저장/닫기 버튼
        self.btn_save = QPushButton("저장")
        self.btn_save.clicked.connect(self._save_lanes)
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(self.btn_save)

        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.reject)
        btn_close.setStyleSheet("padding: 8px 16px;")
        button_layout.addWidget(btn_close)

        layout.addLayout(button_layout)

    def _load_image(self):
        """이미지 로드"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.original_size = (w, h)
            self._update_display()
            self.image_widget.setZoom(0.8)  # 80% 크기로 표시
            self._update_zoom_label()

    def _start_detection(self):
        """SAM3 검출 시작"""
        if self.original_image is None:
            QMessageBox.warning(self, "오류", "이미지가 로드되지 않았습니다.")
            return

        self.btn_detect.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.info_label.setText("SAM3 모델 로딩 및 차선 검출 중...")

        prompts = [p.strip() for p in self.prompt_edit.text().split(",")]
        conf = self.conf_spinbox.value()
        poly_degree = self.smoothing_combo.currentData()

        self.worker = DetectionWorker(
            self.image_path, prompts, conf, poly_degree
        )
        self.worker.finished.connect(self._on_detection_finished)
        self.worker.error.connect(self._on_detection_error)
        self.worker.start()

    def _on_detection_finished(self, lanes):
        """검출 완료 처리"""
        self.detected_lanes = lanes
        self.btn_detect.setEnabled(True)

        if lanes:
            self.btn_save.setEnabled(True)
            self.info_label.setText(f"검출 완료! {len(lanes)}개의 차선을 찾았습니다.")
            self.result_label.setText(f"검출된 차선: {len(lanes)}개")
        else:
            self.info_label.setText("차선을 찾지 못했습니다. 다른 프롬프트를 시도해보세요.")
            self.result_label.setText("검출된 차선: 0개")

        self._update_display()

    def _on_detection_error(self, error_msg):
        """검출 에러 처리"""
        self.btn_detect.setEnabled(True)
        self.info_label.setText(f"검출 오류: {error_msg}")
        QMessageBox.critical(self, "검출 오류", f"SAM3 검출 중 오류 발생:\n{error_msg}")

    def _update_display(self):
        """화면 갱신 - 수동 라벨링과 동일한 방식으로 표시"""
        if self.original_image is None:
            return

        display_img = self.original_image.copy()

        # 검출된 차선 그리기 (수동 라벨링과 동일한 스타일)
        for lane in self.detected_lanes:
            lane_id = lane['id']
            points = lane['points']
            color = self.LANE_COLORS[lane_id % len(self.LANE_COLORS)]

            # 점 그리기
            for px, py in points:
                cv2.circle(display_img, (px, py), 4, color, -1)

            # 선 그리기 (점이 2개 이상이면 연결)
            if len(points) >= 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(display_img, [pts], False, color, 2)

        # Qt 이미지로 변환
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_widget.setPixmap(pixmap)
        self._update_zoom_label()

    def _adjust_zoom(self, delta):
        """줌 조정"""
        current_zoom = self.image_widget.getZoom()
        self.image_widget.setZoom(current_zoom + delta)
        self._update_zoom_label()

    def _fit_to_window(self):
        """창에 맞게 줌"""
        if self.original_size:
            scroll_size = self.scroll_area.viewport().size()
            orig_w, orig_h = self.original_size

            zoom_w = scroll_size.width() / orig_w
            zoom_h = scroll_size.height() / orig_h
            new_zoom = min(zoom_w, zoom_h) * 0.95

            self.image_widget.setZoom(new_zoom)
            self._update_zoom_label()

    def _update_zoom_label(self):
        """줌 레이블 업데이트"""
        zoom = self.image_widget.getZoom()
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _get_save_path(self):
        """저장 경로 반환 (lane_gt/ 폴더)"""
        img_path = Path(self.image_path)
        parts = list(img_path.parts)

        if 'image' in parts:
            idx = parts.index('image')
            parts[idx] = 'lane_gt'
            json_path = Path(*parts).with_suffix('.json')
            return json_path

        return img_path.with_suffix('.json')

    def _save_lanes(self):
        """검출된 차선을 lane_gt 포맷으로 저장"""
        if not self.detected_lanes:
            QMessageBox.warning(self, "저장 오류", "저장할 차선이 없습니다.")
            return

        json_path = self._get_save_path()
        json_path.parent.mkdir(parents=True, exist_ok=True)

        # 저장용 데이터 (mask 제외, points만)
        lanes_to_save = []
        for i, lane in enumerate(self.detected_lanes):
            lanes_to_save.append({
                "id": i,
                "points": lane['points']
            })

        data = {
            "image": Path(self.image_path).name,
            "lanes": lanes_to_save
        }

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            QMessageBox.information(
                self, "저장 완료",
                f"저장됨: {json_path}\n\n{len(lanes_to_save)}개의 차선이 저장되었습니다."
            )
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"저장 실패: {e}")
