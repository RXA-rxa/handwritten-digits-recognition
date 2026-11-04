import sys
from PyQt5.QtWidgets import QApplication, QMainWindow

app = QApplication(sys.argv)
w = QMainWindow()
w.show()
sys.exit(app.exec_())