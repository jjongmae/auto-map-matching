from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QProgressBar, QCheckBox, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter
import cv2
import numpy as np

from app.core.vanishing_point import detect_lines_lsd, compute_vanishing_points, visualize_vanishing_points
try:
    from app.core.segmentation import get_segmentor
except ImportError:
    get_segmentor = None

class VPDetectionThread(QThread):
    finished = Signal(object, object, object, object, object) # vis_img, lines, vps, vps_inliers, mask
    
    def __init__(self, image_path, use_segmentation=False):
        super().__init__()
        self.image_path = image_path
        self.use_segmentation = use_segmentation
        
    def run(self):
        image = cv2.imread(self.image_path)
        if image is None:
            self.finished.emit(None, None, None, None, None)
            return
            
        mask = None
        detect_image = image
        
        # 1. 분할 (활성화된 경우)
        if self.use_segmentation and get_segmentor is not None:
            segmentor = get_segmentor()
            if segmentor:
                mask = segmentor.segment_road(image)
                if mask is not None:
                    # 검출을 위해 이미지에 마스크 적용
                    detect_image = cv2.bitwise_and(image, image, mask=mask)
        
        # 2. 선 검출
        lines = detect_lines_lsd(detect_image)
        
        # 3. 소실점(VP) 계산
        vps, vps_inliers = compute_vanishing_points(lines, image.shape)
        
        # 4. 시각화
        # 기본 시각화
        vis_img = image.copy()
        
        # 마스크가 있으면 오버레이
        if mask is not None:
            # 빨간색 오버레이 생성
            overlay = np.zeros_like(vis_img)
            overlay[:] = (0, 0, 255) # Red
            # 도로는 255. 도로를 강조할까요? 아니면 비도로를?
            # 도로를 투명한 빨간색으로 강조
            vis_img = np.where(mask[..., None] > 0, cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0), vis_img)
        
        vis_img = visualize_vanishing_points(vis_img, lines, vps, vps_inliers)
        
        self.finished.emit(vis_img, lines, vps, vps_inliers, mask)

class ResizableLabel(QLabel):
    """
    위젯 크기에 맞게 픽셀맵을 조정하면서 종횡비를 유지하는 QLabel입니다.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._pixmap = None

    def setPixmap(self, pixmap):
        self._pixmap = pixmap
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
        
        # 이미지 중앙 정렬
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

class VanishingPointDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vanishing Point Detection")
        self.resize(1200, 800)
        
        self.image_path = image_path
        
        # 레이아웃
        layout = QVBoxLayout(self)
        
        # 제어
        controls = QHBoxLayout()
        self.chk_segmentation = QCheckBox("Deep Learning 도로 분할 사용 (SegFormer)")
        self.chk_segmentation.setChecked(True) # 가능한 경우 기본 켜짐?
        if get_segmentor is None:
            self.chk_segmentation.setEnabled(False)
            self.chk_segmentation.setText("Deep Learning 도로 분할 불가 (라이브러리 미설치)")
            
        controls.addWidget(self.chk_segmentation)
        controls.addStretch()
        layout.addLayout(controls)
        
        # 정보 패널
        self.info_label = QLabel("Initializing...")
        self.info_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(self.info_label)
        
        # 이미지 표시 -> ResizableLabel 사용
        self.image_label = ResizableLabel()
        layout.addWidget(self.image_label, stretch=1) # stretch=1로 가용 공간 차지
        
        # 진행률 표시줄
        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # 결정되지 않음
        layout.addWidget(self.progress)
        
        # 닫기 버튼
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
        
        # 체크박스 연결
        self.chk_segmentation.stateChanged.connect(self.start_detection)
        
        # 검출 시작
        self.start_detection()
        
    def start_detection(self):
        self.info_label.setText("처리 중... (첫 실행 시 모델 로드에 시간이 더 걸릴 수 있음)")
        self.image_label.setPixmap(None) # 지우기
        self.progress.show()
        
        use_seg = self.chk_segmentation.isChecked() and self.chk_segmentation.isEnabled()
        
        self.thread = VPDetectionThread(self.image_path, use_segmentation=use_seg)
        self.thread.finished.connect(self.on_detection_finished)
        self.thread.start()
        
    def on_detection_finished(self, vis_img, lines, vps, vps_inliers, mask):
        self.progress.hide()
        
        if vis_img is None:
            self.info_label.setText("Error: Could not load image.")
            return
            
        # 픽셀맵 생성
        h, w, ch = vis_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(vis_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # ResizableLabel에 설정 (자동으로 스케일링 처리)
        self.image_label.setPixmap(pixmap)
        
        # 정보 업데이트
        num_lines = len(lines) if lines is not None else 0
        info_text = f"Total Lines Detected: {num_lines}\n"
        if mask is not None:
             info_text += "분할: 적용됨 (도로 영역 빨간색 강조)\n"
        else:
             info_text += "분할: 꺼짐 (전체 이미지 사용)\n"
             
        for i, vp in enumerate(vps):
            num_inliers = len(vps_inliers[i])
            info_text += f"VP{i+1}: ({vp[0]:.1f}, {vp[1]:.1f}) - 인라이어: {num_inliers}\n"
            
        self.info_label.setText(info_text)
