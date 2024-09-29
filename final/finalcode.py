import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread,pyqtSignal



current_file_dir = os.path.dirname(__file__)
classification_path = os.path.abspath(os.path.join(current_file_dir, '..','classification'))
detection_path = os.path.abspath(os.path.join(current_file_dir, '..', 'detection'))
segmentation_path = os.path.abspath(os.path.join(current_file_dir, '..', 'segmentation'))

# classification 폴더의 파일 확인
ui_file_path = os.path.join(classification_path, 'new1.ui')
ui_file_path2 = os.path.join(detection_path, 'new2.ui')
ui_file_path3 = os.path.join(segmentation_path, 'new3.ui')
print("Classification UI file path:", ui_file_path)
print("Detection UI file path:", ui_file_path2)
print("Segmentation UI file path:", ui_file_path3)

sys.path.append(classification_path)
sys.path.append(detection_path)
sys.path.append(segmentation_path)
print(sys.path)

try:
    import classification_project
    import detection_project
    import segmentation_project
    print("Module imported successfully")
except ImportError as e:
    print("ImportError:", e)

ui_path_result=current_file_dir+'\\new4.ui'
form_class = uic.loadUiType(ui_path_result)[0]

class MainWindow(QTabWidget,form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)  # UI 초기화
        self.classification_tab = self.findChild(QWidget, 'tab')
        self.detection_tab = self.findChild(QWidget, 'tab_2')
        self.segmentation_tab = self.findChild(QWidget, 'tab_3')
        # classification_project의 WindowClass 인스턴스를 생성합니다.
        self.window_class_instance = classification_project.WindowClass()
        self.window_class_instance2 = detection_project.WindowClass()
        self.window_class_instance3 = segmentation_project.WindowClass()
        
        # classification tab
        tab_layout = QVBoxLayout(self.classification_tab)
        self.classification_tab.setLayout(tab_layout)
        tab_layout.addWidget(self.window_class_instance)
        
        # detection tab
        tab_layout = QVBoxLayout(self.detection_tab)
        self.detection_tab.setLayout(tab_layout)
        tab_layout.addWidget(self.window_class_instance2)
        
        # segmentation tab
        tab_layout = QVBoxLayout(self.segmentation_tab)
        self.segmentation_tab.setLayout(tab_layout)
        tab_layout.addWidget(self.window_class_instance3)
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = MainWindow()
    window.setWindowTitle("QTabWidget Example")
    window.show()
    
    sys.exit(app.exec_())