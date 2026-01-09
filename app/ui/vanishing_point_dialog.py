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
        
        # 1. Segmentation (if enabled)
        if self.use_segmentation and get_segmentor is not None:
            segmentor = get_segmentor()
            if segmentor:
                mask = segmentor.segment_road(image)
                if mask is not None:
                    # Apply mask to image for detection
                    detect_image = cv2.bitwise_and(image, image, mask=mask)
        
        # 2. Detect Lines
        lines = detect_lines_lsd(detect_image)
        
        # 3. Compute VPs
        vps, vps_inliers = compute_vanishing_points(lines, image.shape)
        
        # 4. Visualize
        # Base visualization
        vis_img = image.copy()
        
        # Overlay mask if exists
        if mask is not None:
            # Create red overlay
            overlay = np.zeros_like(vis_img)
            overlay[:] = (0, 0, 255) # Red
            # Mask is 255 for road. We want to highlight road? Or non-road?
            # Highlight road with transparent red
            vis_img = np.where(mask[..., None] > 0, cv2.addWeighted(vis_img, 0.7, overlay, 0.3, 0), vis_img)
        
        vis_img = visualize_vanishing_points(vis_img, lines, vps, vps_inliers)
        
        self.finished.emit(vis_img, lines, vps, vps_inliers, mask)

class ResizableLabel(QLabel):
    """
    A QLabel that scales its pixmap to fit the widget size while maintaining aspect ratio.
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
        
        # Center the image
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

class VanishingPointDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vanishing Point Detection")
        self.resize(1200, 800)
        
        self.image_path = image_path
        
        # Layout
        layout = QVBoxLayout(self)
        
        # Controls
        controls = QHBoxLayout()
        self.chk_segmentation = QCheckBox("Deep Learning 도로 분할 사용 (SegFormer)")
        self.chk_segmentation.setChecked(True) # Default on if available?
        if get_segmentor is None:
            self.chk_segmentation.setEnabled(False)
            self.chk_segmentation.setText("Deep Learning 도로 분할 불가 (라이브러리 미설치)")
            
        controls.addWidget(self.chk_segmentation)
        controls.addStretch()
        layout.addLayout(controls)
        
        # Info Panel
        self.info_label = QLabel("Initializing...")
        self.info_label.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f0f0f0;")
        layout.addWidget(self.info_label)
        
        # Image Display -> Use ResizableLabel
        self.image_label = ResizableLabel()
        layout.addWidget(self.image_label, stretch=1) # stretch=1 to take available space
        
        # Progress Bar
        self.progress = QProgressBar()
        self.progress.setRange(0, 0) # Indeterminate
        layout.addWidget(self.progress)
        
        # Close Button
        btn_close = QPushButton("닫기")
        btn_close.clicked.connect(self.accept)
        layout.addWidget(btn_close)
        
        # Connect Checkbox
        self.chk_segmentation.stateChanged.connect(self.start_detection)
        
        # Start Detection
        self.start_detection()
        
    def start_detection(self):
        self.info_label.setText("Processing... (First run may take longer to load model)")
        self.image_label.setPixmap(None) # Clear
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
            
        # Create Pixmap
        h, w, ch = vis_img.shape
        bytes_per_line = ch * w
        qt_image = QImage(vis_img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Set to ResizableLabel (it handles scaling automatically)
        self.image_label.setPixmap(pixmap)
        
        # Update Info
        num_lines = len(lines) if lines is not None else 0
        info_text = f"Total Lines Detected: {num_lines}\n"
        if mask is not None:
             info_text += "Segmentation: Applied (Road Area Highlighted in Red)\n"
        else:
             info_text += "Segmentation: Off (Using Full Image)\n"
             
        for i, vp in enumerate(vps):
            num_inliers = len(vps_inliers[i])
            info_text += f"VP{i+1}: ({vp[0]:.1f}, {vp[1]:.1f}) - Inliers: {num_inliers}\n"
            
        self.info_label.setText(info_text)
