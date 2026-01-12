"""
맵 매칭 UI - 애플리케이션 진입점
"""
import sys
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MapMatcherWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapMatcherWindow()
    window.show()
    sys.exit(app.exec())
