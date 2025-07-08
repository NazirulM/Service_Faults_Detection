import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget
)
from PyQt5.QtGui import QIcon
from real_time_processing import RealTimeProcessing
from main_menu import MainMenu
from local_processing import LocalProcessing
from utils.utility import UtilityFunctions

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AutoServo App")
        self.resize(600, 400)
        self.setWindowIcon(QIcon("LogoFyp.png"))
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.menu = MainMenu()
        self.realtime = RealTimeProcessing(self)
        self.local = LocalProcessing(self)

        self.stack.addWidget(self.menu)
        self.stack.addWidget(self.realtime)
        self.stack.addWidget(self.local)

        self.menu.realtime_clicked.connect(self.goto_realtime)
        self.menu.local_clicked.connect(self.goto_local)

    def goto_realtime(self):
        self.stack.setCurrentWidget(self.realtime)
        print(f"Current widget: {self.stack.currentIndex()}")

    def goto_local(self):
        self.stack.setCurrentWidget(self.local)
        print(f"Current widget: {self.stack.currentIndex()}")

    def show_screen(self, index):
        self.stack.setCurrentIndex(index)

def main():
    app = QApplication(sys.argv)

    app.setStyleSheet(UtilityFunctions.loadStyles(filename="styling/styles.qss"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 