import sys
# import cv2
import numpy as np
# import random
from PyQt5.QtWidgets import QApplication
from tournament_dashboard_copy import TournamentDashboard

# Main application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Apply stylesheet for a modern look
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #f9fafb;
            color: #1f2937;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            margin-top: 12px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 6px 12px;
        }
        QPushButton:hover {
            background-color: #e5e7eb;
        }
        QLabel {
            color: #1f2937;
        }
    """)
    
    window = TournamentDashboard()
    window.show()
    
    sys.exit(app.exec_())