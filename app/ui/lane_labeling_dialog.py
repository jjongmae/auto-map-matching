"""
차선 정답지 생성을 위한 어노테이션 다이얼로그
"""
import json
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QMessageBox, QScrollArea, QWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter
import cv2
import numpy as np


class ZoomableImageWidget(QWidget):
    """
    확대/축소 및 클릭 이벤트를 지원하는 이미지 위젯.
    QScrollArea 내에서 사용됩니다.
    """
    clicked = Signal(int, int)  # 원본 이미지 좌표 (x, y)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._original_size = None
        self._zoom = 1.0

    def setPixmap(self, pixmap, original_size=None):
        self._pixmap = pixmap
        self._original_size = original_size
        self._update_size()

    def setZoom(self, zoom):
        self._zoom = max(0.1, min(zoom, 10.0))  # 0.1x ~ 10x
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
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        painter.drawPixmap(0, 0, scaled)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._pixmap and self._original_size:
            # 클릭 위치를 원본 이미지 좌표로 변환
            click_x = event.pos().x()
            click_y = event.pos().y()

            # 줌 레벨 반영하여 원본 좌표 계산
            orig_w, orig_h = self._original_size
            display_w = self._pixmap.width() * self._zoom
            display_h = self._pixmap.height() * self._zoom

            img_x = int(click_x * orig_w / display_w)
            img_y = int(click_y * orig_h / display_h)

            # 범위 체크
            if 0 <= img_x < orig_w and 0 <= img_y < orig_h:
                self.clicked.emit(img_x, img_y)


class LaneLabelingDialog(QDialog):
    # 차선별 색상 (최대 10개)
    LANE_COLORS = [
        (255, 0, 0),      # 빨강
        (0, 255, 0),      # 초록
        (0, 0, 255),      # 파랑
        (255, 255, 0),    # 노랑
        (255, 0, 255),    # 마젠타
        (0, 255, 255),    # 시안
        (255, 128, 0),    # 주황
        (128, 0, 255),    # 보라
        (0, 255, 128),    # 민트
        (255, 128, 128),  # 연분홍
    ]

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("차선 라벨링")
        self.resize(1400, 900)

        self.image_path = image_path
        self.original_image = None
        self.original_size = None

        # 차선 데이터: [{"id": 0, "points": [[x, y], ...]}, ...]
        self.lanes = []
        self.current_lane_id = 0

        self._setup_ui()
        self._load_image()
        self._load_existing_annotation()
        self._update_display()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 상단 정보
        self.info_label = QLabel()
        self.info_label.setStyleSheet(
            "font-size: 14px; padding: 10px; background-color: #f0f0f0; border-radius: 5px;"
        )
        layout.addWidget(self.info_label)

        # 스크롤 영역 + 이미지 위젯
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        self.image_widget = ZoomableImageWidget()
        self.image_widget.clicked.connect(self._on_image_clicked)
        self.scroll_area.setWidget(self.image_widget)

        layout.addWidget(self.scroll_area, stretch=1)

        # 단축키 안내
        shortcut_label = QLabel(
            "[클릭] 점 추가 | [Enter] 다음 차선 | [Backspace] 점 삭제 | "
            "[Delete] 차선 삭제 | [S] 저장 | [R] 초기화"
        )
        shortcut_label.setStyleSheet(
            "font-size: 12px; padding: 8px; background-color: #e0e0e0; border-radius: 3px;"
        )
        shortcut_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(shortcut_label)

        # 하단 버튼
        button_layout = QHBoxLayout()

        btn_prev_lane = QPushButton("◀ 이전 차선")
        btn_prev_lane.clicked.connect(self._prev_lane)
        button_layout.addWidget(btn_prev_lane)

        btn_next_lane = QPushButton("다음 차선 ▶")
        btn_next_lane.clicked.connect(self._next_lane)
        button_layout.addWidget(btn_next_lane)

        button_layout.addStretch()

        # 줌 버튼
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

        btn_save = QPushButton("저장 (S)")
        btn_save.clicked.connect(self._save_annotation)
        button_layout.addWidget(btn_save)

        btn_reset = QPushButton("초기화 (R)")
        btn_reset.clicked.connect(self._reset_annotation)
        button_layout.addWidget(btn_reset)

        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        button_layout.addWidget(btn_close)

        layout.addLayout(button_layout)

    def _load_image(self):
        """이미지 로드"""
        self.original_image = cv2.imread(self.image_path)
        if self.original_image is not None:
            h, w = self.original_image.shape[:2]
            self.original_size = (w, h)

    def _get_annotation_path(self):
        """정답지 JSON 파일 경로 반환"""
        img_path = Path(self.image_path)
        parts = list(img_path.parts)

        if 'image' in parts:
            idx = parts.index('image')
            parts[idx] = 'lane_gt'
            json_path = Path(*parts).with_suffix('.json')
            return json_path

        return img_path.with_suffix('.json')

    def _load_existing_annotation(self):
        """기존 정답지 불러오기"""
        json_path = self._get_annotation_path()

        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.lanes = data.get('lanes', [])

                if self.lanes:
                    self.current_lane_id = max(lane['id'] for lane in self.lanes)
            except Exception as e:
                print(f"정답지 로드 오류: {e}")
                self.lanes = []

    def _save_annotation(self):
        """정답지 저장"""
        json_path = self._get_annotation_path()
        json_path.parent.mkdir(parents=True, exist_ok=True)

        lanes_to_save = [lane for lane in self.lanes if lane['points']]

        # ID 재할당 (입력 순서대로 0, 1, 2...)
        for i, lane in enumerate(lanes_to_save):
            lane['id'] = i

        data = {
            'image': Path(self.image_path).name,
            'lanes': lanes_to_save
        }

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.lanes = lanes_to_save
            if lanes_to_save:
                self.current_lane_id = lanes_to_save[-1]['id']
            else:
                self.current_lane_id = 0

            self._update_display()
            QMessageBox.information(self, "저장 완료", f"저장됨: {json_path}")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"저장 실패: {e}")

    def _reset_annotation(self):
        """모든 어노테이션 초기화"""
        reply = QMessageBox.question(
            self, "초기화 확인",
            "모든 차선 데이터를 삭제하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.lanes = []
            self.current_lane_id = 0
            self._update_display()

    def _get_current_lane(self):
        """현재 차선 데이터 반환 (없으면 생성)"""
        for lane in self.lanes:
            if lane['id'] == self.current_lane_id:
                return lane

        new_lane = {'id': self.current_lane_id, 'points': []}
        self.lanes.append(new_lane)
        return new_lane

    def _on_image_clicked(self, x, y):
        """이미지 클릭 시 점 추가"""
        lane = self._get_current_lane()
        lane['points'].append([x, y])
        self._update_display()

    def _next_lane(self):
        """다음 차선으로 이동"""
        self.current_lane_id += 1
        self._update_display()

    def _prev_lane(self):
        """이전 차선으로 이동"""
        if self.current_lane_id > 0:
            self.current_lane_id -= 1
            self._update_display()

    def _delete_last_point(self):
        """현재 차선의 마지막 점 삭제"""
        for lane in self.lanes:
            if lane['id'] == self.current_lane_id and lane['points']:
                lane['points'].pop()
                self._update_display()
                return

    def _delete_current_lane(self):
        """현재 차선 삭제"""
        self.lanes = [lane for lane in self.lanes if lane['id'] != self.current_lane_id]
        self._update_display()

    def _adjust_zoom(self, delta):
        """줌 레벨 조정"""
        current_zoom = self.image_widget.getZoom()
        new_zoom = current_zoom + delta
        self.image_widget.setZoom(new_zoom)
        self._update_zoom_label()

    def _fit_to_window(self):
        """창에 맞게 줌 조정"""
        if self.original_size:
            scroll_size = self.scroll_area.viewport().size()
            orig_w, orig_h = self.original_size

            zoom_w = scroll_size.width() / orig_w
            zoom_h = scroll_size.height() / orig_h
            new_zoom = min(zoom_w, zoom_h) * 0.95  # 약간 여유

            self.image_widget.setZoom(new_zoom)
            self._update_zoom_label()

    def _update_zoom_label(self):
        """줌 레벨 표시 업데이트"""
        zoom = self.image_widget.getZoom()
        self.zoom_label.setText(f"{int(zoom * 100)}%")

    def _update_display(self):
        """화면 갱신"""
        if self.original_image is None:
            return

        display_img = self.original_image.copy()

        # 모든 차선 그리기
        for lane in self.lanes:
            lane_id = lane['id']
            points = lane['points']
            color = self.LANE_COLORS[lane_id % len(self.LANE_COLORS)]

            thickness = 2 if lane_id == self.current_lane_id else 1
            point_radius = 4 if lane_id == self.current_lane_id else 3

            for px, py in points:
                cv2.circle(display_img, (px, py), point_radius, color, -1)

            if len(points) >= 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(display_img, [pts], False, color, thickness)

        # Qt 이미지로 변환
        h, w, ch = display_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(display_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)

        self.image_widget.setPixmap(pixmap, self.original_size)
        self._update_zoom_label()

        # 정보 업데이트
        current_lane = None
        for lane in self.lanes:
            if lane['id'] == self.current_lane_id:
                current_lane = lane
                break

        current_points = len(current_lane['points']) if current_lane else 0
        total_lanes = len([l for l in self.lanes if l['points']])

        color = self.LANE_COLORS[self.current_lane_id % len(self.LANE_COLORS)]
        color_name = f"RGB({color[0]}, {color[1]}, {color[2]})"

        self.info_label.setText(
            f"현재 차선: {self.current_lane_id} ({color_name}) | "
            f"점 개수: {current_points} | "
            f"총 차선 수: {total_lanes}"
        )

    def keyPressEvent(self, event):
        """키보드 단축키 처리"""
        key = event.key()

        if key == Qt.Key_Return or key == Qt.Key_Enter:
            self._next_lane()
        elif key == Qt.Key_Backspace:
            self._delete_last_point()
        elif key == Qt.Key_Delete:
            self._delete_current_lane()
        elif key == Qt.Key_S:
            self._save_annotation()
        elif key == Qt.Key_R:
            self._reset_annotation()
        elif key == Qt.Key_Plus or key == Qt.Key_Equal:
            self._adjust_zoom(0.25)
        elif key == Qt.Key_Minus:
            self._adjust_zoom(-0.25)
        else:
            super().keyPressEvent(event)
