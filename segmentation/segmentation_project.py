import sys, os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread, pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import segmentation_train
from matplotlib.figure import Figure
from PIL import Image,ImageOps
import torchvision.transforms as T

current_file_dir = os.path.dirname(__file__)

# UI 파일의 절대 경로 설정
ui_file_path = os.path.join(current_file_dir, 'new3.ui')

if not os.path.isfile(ui_file_path):
    print(f"UI file does not exist: {ui_file_path}")
else:
    # UI 파일 연결
    form_class = uic.loadUiType(ui_file_path)[0]


class TrainThread(QThread):
    progress = pyqtSignal(str)  # Progress signal to update the GUI
    
    def __init__(self, args, window):
        super().__init__()
        #classification
        self.args = args
        self.window = window

    def run(self):
        original_stdout = sys.stdout  # Save a reference to the original standard output
        sys.stdout = OutputCapture(self.progress)  # Redirect stdout to capture output
        segmentation_train.main(self.window, self.args)
        sys.stdout = original_stdout


class TrainThread_test(QThread):
    progress = pyqtSignal(str)  # Progress signal to update the GUI
    
    def __init__(self,window):
        super().__init__()
        self.window=window
        self._running=True

    def run(self):
        original_stdout = sys.stdout  # 원래의 표준 출력(터미널)을 저장
        # 표준 출력을 OutputCapture로 리다이렉트하여 터미널 출력을 캡처
        sys.stdout = OutputCapture(self.progress) 
        # 학습 작업 시작
        self.window.test()
        # 학습 작업이 끝나면 표준 출력을 원래대로 복구
        sys.stdout = original_stdout       
# 터미널 class
class OutputCapture:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        if text.strip():  # Only emit non-empty text
            self.signal.emit(text)

    def flush(self):
        pass

# 화면을 띄우는데 사용되는 Class 선언
class WindowClass(QTabWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # QLabel에서 텍스트가 길어지면 자동으로 줄바꿈
        self.file_dirShow.setWordWrap(True)

        
        #segmentation
        self.data_directory.clicked.connect(lambda:self.openFolderDialog('data_path'))
        self.out_directory.clicked.connect(lambda:self.openFolderDialog('output_path'))
        self.model_teach.clicked.connect(self.run_command)
        self.model_teach_exit.clicked.connect(self.for_key)
        
        self.key=False
        
        self.model_directory.clicked.connect(lambda:self.openFolderDialog("model_path"))
        self.test_directory.clicked.connect(lambda:self.openFolderDialog("test_folder_path"))
        self.training.clicked.connect(self.thread_open)
        
        # 훈련 그래프
        self.figure2 = Figure()
        self.canvas2 = FigureCanvas(self.figure2)
        acc_widget2 = self.findChild(QWidget, 'graph_canvas')
        layout2 = QVBoxLayout(acc_widget2)
        layout2.addWidget(self.canvas2)
        
        self.ax1 = self.figure2.add_subplot(2, 1, 1)
        self.ax2 = self.figure2.add_subplot(2, 1, 2)
        
        self.ax1.set_title('Loss')
        self.ax1.set_xlim(0, 50)  # x축 범위 설정
        self.ax1.set_ylim(0, 1)   # y축 범위 설정

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlim(0, 50)  # x축 범위 설정
        self.ax2.set_ylim(0, 1)   # y축 범위 설정
        
        # test 그래프
        self.test_figure = Figure()
        self.test_canvas = FigureCanvas(self.test_figure)
        
        # UI에서 graphWidget을 가져와 layout 설정
        test_layout = QVBoxLayout(self.train_image)  # graphWidget이 있는 QFrame 또는 QWidget
        test_layout.addWidget(self.test_canvas)
        
    def thread_open(self):
        self.thread = TrainThread_test(self)
        self.thread.progress.connect(self.display_test_output)  # TrainThread의 출력을 GUI에 연결
        self.thread.start() 
        
    def display_test_output(self, text):
        self.segmentation_shell_2.append(text)  # 터미널 출력을 텍스트 위젯에 추가합니다.
    
    def run_shell_command(self):
        self.segmentation_shell.clear()  # 텍스트 위젯을 비웁니다.
        self.worker.start()    # 스레드를 시작하여 명령어를 실행합니다.

    def display_output(self, text):
        self.segmentation_shell.append(text)  # 터미널 출력을 텍스트 위젯에 추가합니다.
    
    def plot(self,x_arr,to_numpy_valid,to_numpy_train):
        self.figure2.clear()
        print("project_plot_printing")
        print("last train_loss", to_numpy_train[0][-1])
        print("last valid_loss", to_numpy_valid[0][-1])
        # Create subplots
        self.ax1 = self.figure2.add_subplot(2, 1, 1)
        self.ax1.plot(x_arr, to_numpy_train[0], '-', label='Train loss')
        self.ax1.plot(x_arr, to_numpy_valid[0], '--', label='Valid loss')
        handles, labels = self.ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax1.legend(by_label.values(), by_label.keys())
        
        self.ax2 = self.figure2.add_subplot(2, 1, 2)
        self.ax2.plot(x_arr, to_numpy_train[1], '-', label='Train acc')
        self.ax2.plot(x_arr, to_numpy_valid[1], '--', label='Valid acc')
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax2.legend(by_label.values(), by_label.keys())
        
        self.ax1.set_title('Loss')
        self.ax2.set_title('Accuracy')
        # D222raw the canvas
        self.canvas2.draw()

    
    def for_key(self):
        self.key=True
    
    def stop_training(self):
        if hasattr(self, 'thread') and self.thread.isRunning() and self.key:
            return False
        return True
    
    #classification
    def openFolderDialog(self,path_type):
        folder_path=QFileDialog.getExistingDirectory(self, "폴더 선택", "")
        if folder_path:
            if path_type == "data_path":
                self.folder_path = folder_path
                self.file_dirShow.setText(f"Select Test File path: {self.folder_path}")
            elif path_type == "output_path":
                self.resultfolder_path = folder_path
                if(self.folder_path==""):
                    self.file_dirShow.setText(f"input: File not selected \noutput: {self.resultfolder_path}")  
                else:
                    self.file_dirShow.setText(f"input: {self.folder_path}\noutput: {self.resultfolder_path}")
            elif path_type == "model_path":
                self.model_path = folder_path
                self.file_dirShow2.setText(f"Select Test File path: {self.model_path}")
            elif path_type == "test_folder_path":
                self.test_folder_path = folder_path
                if(self.model_path==""):
                    self.file_dirShow2.setText(f"input: File not selected \noutput: {self.test_folder_path}")  
                else:
                    self.file_dirShow2.setText(f"input: {self.model_path}\noutput: {self.test_folder_path}")
    def openFileDialog(self,path_type):
        file_path, _ = QFileDialog.getOpenFileName(self, "파일 선택", "", "모든 파일 (*);;텍스트 파일 (*.txt)")
        if file_path:
            if path_type == "model_path":
                self.model_path = file_path
                self.file_dirShow2.setText(f"Select Test File path: {self.model_path}")
            elif path_type == "img_path":
                self.img_path = file_path
                if(self.model_path==""):
                    self.file_dirShow2.setText(f"input: File not selected \noutput: {self.img_path}")  
                else:
                    self.file_dirShow2.setText(f"input: {self.model_path}\noutput: {self.img_path}")   
    def run_command(self):
        self.key = False
        self.epochs_spinBox = self.findChild(QSpinBox, 'epochs_spinBox') 
        epoch=self.epochs_spinBox.value()
        # worker
        self.worker_spinBox = self.findChild(QSpinBox, 'worker_spinBox') 
        worker=self.worker_spinBox.value()
        # learning rate
        self.lr_spinBox = self.findChild(QDoubleSpinBox, 'lr_spinBox') 
        lr=self.lr_spinBox.value()
        # model
        self.model_comboBox = self.findChild(QComboBox, 'model_comboBox') 
        model=self.model_comboBox.currentText()
        # weight
        self.weight_comboBox = self.findChild(QComboBox, 'weight_comboBox') 
        weight=self.weight_comboBox.currentText()
        # device
        self.device_comboBox = self.findChild(QComboBox, 'device_comboBox') 
        device=self.device_comboBox.currentText()
        #new append
        #batch-size
        self.batch_size_spinBox = self.findChild(QSpinBox, 'batch_size_spinBox')
        batch_size = self.batch_size_spinBox.value()
        #dateset
        self.dataset_comboBox = self.findChild(QComboBox, 'dataset_comboBox')
        dataset = self.dataset_comboBox.currentText()
        
        if self.resultfolder_path and self.folder_path:
            args = segmentation_train.get_args_parser(self.folder_path, epoch, worker, lr, model, weight, device, self.resultfolder_path, batch_size, dataset)
            self.thread = TrainThread(args, self)
            self.thread.progress.connect(self.display_output)  # TrainThread의 출력을 GUI에 연결
            self.thread.start() 
    def decode_segmap(self,image, nc=21): 
        label_colors = segmentation_train.np.array([(0, 0, 0),  # 0=background
                        # 1=ceiling, 2=chair, 3=door, 4=floor, 5=glassdoor
                        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                        # 6=table, 7=wall, 8=window, 9=chair, 
                        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])
        r = segmentation_train.np.zeros_like(image).astype(segmentation_train.np.uint8)
        g = segmentation_train.np.zeros_like(image).astype(segmentation_train.np.uint8)
        b = segmentation_train.np.zeros_like(image).astype(segmentation_train.np.uint8) 
        for l in range(0, nc):
            idx = image == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]   
        rgb = segmentation_train.np.stack([r, g, b], axis=2)
        return rgb
    
    # def test(self):    
    #     # load pth model

    #     model = train.torch.load(self.model_path)

    #     # set model to inference mode

    #     model.eval()

    #     #print(model)
    #     # prediction

    #     img_path = self.img_path

    #     img = Image.open(img_path)

    #     transform = T.Compose([T.Resize(520),
    #                 T.CenterCrop(480),
    #                 T.ToTensor(),
    #                 T.Normalize(        
    #                 mean=[0.485, 0.456, 0.406],
    #                 std=[0.229, 0.224, 0.225])
    #                 ])
    #     trans_img = transform(img).permute(1, 2, 0)
    #     img = transform(img).unsqueeze(0)
    #     out = model(img)['out']
    #     print(img.shape)
    #     print(out.shape)
    #     om = train.torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    #     print (om.shape)
    #     print (train.np.unique(om))
    #     rgb = self.decode_segmap(om)
    #     self.test_show(trans_img,rgb)
    #     # self.test_figure = Figure()
    #     # self.test_canvas = FigureCanvas(self.test_figure)
        
    #     # # UI에서 graphWidget을 가져와 layout 설정
    #     # test_layout = QVBoxLayout(self.train_image)  # graphWidget이 있는 QFrame 또는 QWidget
    #     # test_layout.addWidget(self.test_canvas)
        
        
    #     # self.test_figure.clear()  # 이전 그래프 지우기
    #     # ax1 = self.test_figure.add_subplot(121)
    #     # ax2 = self.test_figure.add_subplot(122)

    #     # ax1.axis('off')
    #     # ax1.imshow(trans_img)

    #     # ax2.axis('off')
    #     # ax2.imshow(rgb)

    #     # 캔버스 업데이트
    #     # self.test_canvas.draw()    
    # def test_show(self,trans_img,rgb):
    #     self.test_figure.clear()  # 이전 그래프 지우기
    #     ax1 = self.test_figure.add_subplot(121)
    #     ax2 = self.test_figure.add_subplot(122)

    #     ax1.axis('off')
    #     ax1.imshow(trans_img)

    #     ax2.axis('off')
    #     ax2.imshow(rgb)

    #     # 캔버스 업데이트
    #     self.test_canvas.draw()       
    def test(self):
        # load pth model
        model = segmentation_train.torch.load(f'{self.model_path}\\model.pth')
        # set model to inference mode
        model.eval()
        #print(model)

        folder_path = self.test_folder_path
        for filename in os.listdir(folder_path):  # 폴더 내의 모든 파일에 대해 반복
            if filename.endswith(".jpg") or filename.endswith(".jfif") or filename.endswith(".webp"):  # 지원하는 파일 형식 검사
                image_path = os.path.join(folder_path, filename)  # 파일 경로 생성
                img = Image.open(image_path)  # 이미지 파일 열기
                start_time = segmentation_train.time.time()
                img = ImageOps.exif_transpose(img)
                transform = T.Compose([T.Resize(520),
                        T.CenterCrop(480),
                        T.ToTensor(),
                        T.Normalize(        
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
                        ])
                trans_img = transform(img).permute(1, 2, 0)
                img = transform(img).unsqueeze(0)
                img = img.to('cpu')
                out = model(img)['out']
                print(img.shape)
                print(out.shape)

                om = segmentation_train.torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
                end_time = segmentation_train.time.time()
                print (om.shape)
                print (segmentation_train.np.unique(om))
                rgb = self.decode_segmap(om)
                print("prediction one image : ", round(end_time - start_time, 3))
                self.test_show(trans_img,rgb)
                # plt.subplot(121), plt.axis('off'), plt.imshow(trans_img)
                # plt.subplot(122), plt.axis('off'), plt.imshow(rgb)
                # plt.show()    
    def test_show(self,trans_img,rgb):
        self.test_figure.clear()  # 이전 그래프 지우기
        ax1 = self.test_figure.add_subplot(121)
        ax2 = self.test_figure.add_subplot(122)

        ax1.axis('off')
        ax1.imshow(trans_img)

        ax2.axis('off')
        ax2.imshow(rgb)

        # 캔버스 업데이트
        self.test_canvas.draw() 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # WindowClass의 인스턴스 생성
    myWindow = WindowClass()
    # 프로그램 화면을 보여주는 코드
    myWindow.show()
    # 프로그램을 이벤트 루프로 진입시키는(프로그램을 작동시키는) 코드
    sys.exit(app.exec_())   