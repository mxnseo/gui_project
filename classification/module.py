import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QLabel, QFileDialog, QApplication, QSpinBox, QDoubleSpinBox, QComboBox
from PyQt5.QtGui import QPixmap
from PyQt5 import uic
import classification.train as train

# UI 파일 연결
form_class = uic.loadUiType("untitled.ui")[0]

class WindowClass(QTabWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.file_dirShow.setWordWrap(True)
        self.file_directory.clicked.connect(self.openFolderDialog)
        self.model_teach.clicked.connect(self.run_command)
        
        self.accuracy_layout = self.findChild(QVBoxLayout, 'accuracy')
        self.image_label = self.findChild(QLabel, 'image_label')
        
        self.figure = plt.figure(figsize=(10, 5))
        self.al = self.figure.add_subplot(2, 1, 1)
        self.ar = self.figure.add_subplot(2, 1, 2)
        
        self.canvas = FigureCanvas(self.figure)
        self.accuracy_layout.addWidget(self.canvas)
        
    def plot(self, to_numpy_train, to_numpy_valid, epoch):
        self.figure.clear()
        x_arr = np.arange(epoch + 1)
        
        self.al = self.figure.add_subplot(2, 1, 1)
        self.al.plot(x_arr, to_numpy_train[0], '-o', label='Train loss')
        self.al.plot(x_arr, to_numpy_valid[0], '-->', label='Valid loss')
        self.al.set_xlabel('Epoch', size=15)
        self.al.set_ylabel('Loss', size=15)
        self.al.legend()
        
        self.ar = self.figure.add_subplot(2, 1, 2)
        self.ar.plot(x_arr, to_numpy_train[1], '-o', label='Train acc')
        self.ar.plot(x_arr, to_numpy_valid[1], '-->', label='Valid acc')
        self.ar.set_xlabel('Epoch', size=15)
        self.ar.set_ylabel('Accuracy', size=15)
        self.ar.legend()
        
        self.canvas.draw()

        img_path = 'img.jpg'
        self.figure.savefig(img_path)
        self.update_image(img_path)

    def update_image(self, img_path):
        pixmap = QPixmap(img_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size()))

    def openFolderDialog(self):
        self.folder_path = QFileDialog.getExistingDirectory(self, "폴더 선택", "")
        if self.folder_path:
            self.file_dirShow.setText(f"선택된 파일 경로: {self.folder_path}")

    def openFolderDialog_result(self):
        self.resultfolder_path = QFileDialog.getExistingDirectory(self, "폴더 선택", "")

    def run_command(self):
        self.openFolderDialog_result()
        
        epoch = self.findChild(QSpinBox, 'epochs_spinBox').value()
        worker = self.findChild(QSpinBox, 'worker_spinBox').value()
        lr = self.findChild(QDoubleSpinBox, 'lr_spinBox').value()
        model = self.findChild(QComboBox, 'model_comboBox').currentText()
        weight = self.findChild(QComboBox, 'weight_comboBox').currentText()
        
        args = train.get_args_parser(self.folder_path, epoch, worker, lr, model, weight, self.resultfolder_path)
        
        result = train.main(args)
        
        if result is None:
            print("train.main()에서 반환된 결과가 없습니다.")
            return
        
        to_numpy_train, to_numpy_valid = result
        
        if epoch != 0:
            self.plot(to_numpy_train, to_numpy_valid, epoch)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = WindowClass()
    myWindow.show()
    sys.exit(app.exec_())
