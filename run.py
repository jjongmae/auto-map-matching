"""
Map Matching UI - Application Entry Point
"""
import sys
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MapMatcherWindow


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MapMatcherWindow()
    window.show()
    sys.exit(app.exec())
