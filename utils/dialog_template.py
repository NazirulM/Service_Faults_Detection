from PyQt5.QtWidgets import QDialog, QLabel, QVBoxLayout, QDialogButtonBox
from PyQt5.QtGui import QIcon

# ACCEPT OR REJECT DIALOG
class CustomDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AutoServo")
        self.setMinimumSize(1000, 800)
        self.setWindowIcon(QIcon("LogoFyp.png"))

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        layout = QVBoxLayout()
        message = QLabel("Something happened, is that OK?")
        layout.addWidget(message)
        layout.addWidget(self.buttonBox)
        self.setLayout(layout)