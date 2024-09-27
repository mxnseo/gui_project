import sys
import os
from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread,pyqtSignal
from PIL import Image
import cv2
import detection_train
import time
import numpy as np
import torch
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import torchvision.transforms as T

# FutureWarning 무시
#warnings.simplefilter(action='ignore', category=FutureWarning)

current_file_dir = os.path.dirname(__file__)

# UI 파일의 절대 경로 설정
ui_file_path = os.path.join(current_file_dir, 'new2.ui')

if not os.path.isfile(ui_file_path):
    print(f"UI file does not exist: {ui_file_path}")
else:
    # UI 파일 연결
    form_class = uic.loadUiType(ui_file_path)[0]


# Thread 클래스
class TrainThread(QThread):
    progress = pyqtSignal(str)  # Progress signal to update the GUI
    
    def __init__(self, args,window):
        super().__init__()
        self.args = args
        self.window=window
        self._running=True

    def run(self):
        original_stdout = sys.stdout  # 원래의 표준 출력(터미널)을 저장
        # 표준 출력을 OutputCapture로 리다이렉트하여 터미널 출력을 캡처
        sys.stdout = OutputCapture(self.progress) 
        # 학습 작업 시작
        detection_train.main(self.window, self.args)
        # 학습 작업이 끝나면 표준 출력을 원래대로 복구
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

    
# 터미널 출력을 캡처하고 pyqt 시그널로 전송하는 클래스
class OutputCapture:
    def __init__(self, signal):
        self.signal = signal

    def write(self, text):
        # 공백이 아닌 경우에만 시그널로 전송
        if text.strip():  
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
        
        
            
        # dataset directory 버튼 클릭시 openFileDialog 메서드를 호출
        self.data_directory.clicked.connect(lambda:self.openFolderDialog('data_path'))
        # output directory 버튼 클릭시 openFileDialog 메서드를 호출
        self.out_directory.clicked.connect(lambda:self.openFolderDialog('output_path'))
        # 모델학습 버튼 클릭시 run_command 메서드 호출
        self.model_teach.clicked.connect(self.run_command)
        self.model_teach_stop.clicked.connect(self.for_key)
        
        
        self.model_directory.clicked.connect(lambda:self.openFolderDialog("model_path"))
        self.test_directory.clicked.connect(lambda:self.openFolderDialog("test_folder_path"))
        self.training.clicked.connect(self.thread_open)
        
        self.key=False
            
        #그래프
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        acc_widget = self.findChild(QWidget, 'graph_canvas')
        canvas_layout = QVBoxLayout(acc_widget)
        canvas_layout.addWidget(self.canvas)

        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax2 = self.figure.add_subplot(2, 1, 2)
        
        self.ax1.set_title('Loss')
        self.ax1.set_xlim(0, 50)  # x축 범위 설정
        self.ax1.set_ylim(0, 1)   # y축 범위 설정

        self.ax2.set_title('Accuracy')
        self.ax2.set_xlim(0, 50)  # x축 범위 설정
        self.ax2.set_ylim(0, 1)   # y축 범위 설정
        
        # 추론 그래프
        self.test_figure=Figure(figsize=(5, 4), dpi=100)
        self.test_canvas = FigureCanvas(self.test_figure)
        test_widget = self.findChild(QWidget, 'train_image')
        test_canvas = QVBoxLayout(test_widget)
        test_canvas.addWidget(self.test_canvas)
    
    def thread_open(self):
        self.thread = TrainThread_test(self)
        self.thread.progress.connect(self.display_test_output)  # TrainThread의 출력을 GUI에 연결
        self.thread.start() 
        
    def display_test_output(self, text):
        self.detection_shell_2.append(text)  # 터미널 출력을 텍스트 위젯에 추가합니다.       
    def run_shell_command(self):
        self.detection_shell.clear()  # 텍스트 위젯을 비웁니다.
        self.worker.start()    # 스레드를 시작하여 명령어를 실행합니다.

    def display_output(self,text):
        self.detection_shell.append(text)  # 터미널 출력을 텍스트 위젯에 추가합니다.
        
    # 그래프 그리기
    def plot(self,x_arr,to_numpy_valid,to_numpy_train):
        self.figure.clear()
        
        # Create subplots
        self.ax1 = self.figure.add_subplot(2, 1, 1)
        self.ax1.plot(x_arr, to_numpy_train[0], '-', label='Train loss',marker='o')
        self.ax1.plot(x_arr, to_numpy_valid[0], '--', label='Valid loss',marker='o')
        handles, labels = self.ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax1.legend(by_label.values(), by_label.keys())
        
        self.ax2 = self.figure.add_subplot(2, 1, 2)
        self.ax2.plot(x_arr, to_numpy_train[1], '-', label='Train acc',marker='o')
        self.ax2.plot(x_arr, to_numpy_valid[1], '--', label='Valid acc',marker='o')
        handles, labels = self.ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax2.legend(by_label.values(), by_label.keys())
        
        self.ax1.set_title('Loss')
        self.ax2.set_title('Accuracy')
        # canvas 그리기
        self.canvas.draw()
    
    def for_key(self):
        self.key=True
    
    def stop_training(self):
        if hasattr(self, 'thread') and self.thread.isRunning() and self.key:
            return False
        return True

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

    def run_command(self):
        
        # 실행할 명령어 정의
        self.key=False
        # 에포크
        self.epochs_spinBox = self.findChild(QSpinBox, 'epochs_spinBox') 
        epoch=self.epochs_spinBox.value()
        # --aspect-ratio-group-factor
        self.aspect_spinBox = self.findChild(QSpinBox, 'aspect_spinBox') 
        aspect=self.aspect_spinBox.value()
        # learning rate
        self.dataset_comboBox = self.findChild(QComboBox, 'dataset_comboBox') 
        dataset=self.dataset_comboBox.currentText()
        # model
        self.model_comboBox = self.findChild(QComboBox, 'model_comboBox') 
        model=self.model_comboBox.currentText()
        # weight
        self.weight_comboBox = self.findChild(QComboBox, 'weight_comboBox') 
        weight=self.weight_comboBox.currentText()
        # device
        self.device_comboBox = self.findChild(QComboBox, 'device_comboBox') 
        device=self.device_comboBox.currentText()
        
        if(self.resultfolder_path!="" and self.folder_path!=""):
            args = detection_train.get_args_parser(self.folder_path,epoch,aspect,dataset,model,weight,device,self.resultfolder_path)
            self.thread = TrainThread(args, self)
            self.thread.progress.connect(self.display_output)  # TrainThread의 출력을 GUI에 연결
            self.thread.start() 
            # self.thread.start()

    
    def test(self):
        # 모델 로드
        print(f'{self.model_path}/model.pth')
        model = torch.load(f'{self.model_path}/model.pth')
        model.eval()
        
        COCO_INSTANCE_CATEGORY_NAMES = [
            '__background__', 'big robot', 'small robot'
        ]
        test_path = self.test_folder_path
        threshold = 0.2
        total_time = 0  # 전체 테스트 시간 저장
        print("테스트 경로 : ", str(test_path))
        for filename in os.listdir(test_path):
            print("filename : ", filename)
            img_path = os.path.join(test_path, filename)
            img = Image.open(img_path)
            # 시작 시간 기록
            start_time = time.time()
            transform = T.Compose([T.ToTensor()])
            img_tensor = transform(img)
            pred = model([img_tensor])
            
            # 종료 시간 기록
            end_time = time.time()

            # 소요 시간 계산
            elapsed_time = end_time - start_time
            total_time += elapsed_time
            print(f"이미지 {filename} 처리에 걸린 시간: {elapsed_time:.4f}초")

            pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].numpy())]
            pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]
            pred_score = list(pred[0]['scores'].detach().numpy())

            pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
            print("리스트 : ", pred_t)
            print("Point A")
            if pred_t:
                print("Point B")
                pred_t = pred_t[-1]  # 마지막 임계값 초과 인덱스
                pred_boxes = pred_boxes[:pred_t + 1]
                pred_class = pred_class[:pred_t + 1]

                # 이미지 읽기 및 색상 변환
                img_cv = cv2.imread(img_path)
                print("이미지 경로 : ", img_path)
                if os.path.exists(img_path):
                    print("이미지 파일이 존재합니다.")
                if img_cv is None:
                    print("이미지를 로드할 수 없습니다.")
                else:
                    print("이미지 호출 완료")
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                #cv2.imshow('str', img_cv)

                # 사각형 및 텍스트 그리기
                for i in range(len(pred_boxes)):
                    cv2.rectangle(img_cv, 
                        (int(pred_boxes[i][0][0]), int(pred_boxes[i][0][1])),
                        (int(pred_boxes[i][1][0]), int(pred_boxes[i][1][1])), 
                        (0, 255, 0), thickness=3)
                    cv2.putText(img_cv, pred_class[i] + ':' + f'{pred_score[i]:.4f}', 
                        (int(pred_boxes[i][0][0]), int(pred_boxes[i][0][1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)
            
            self.test_show(img_cv)
        print(f"총 테스트 소요 시간: {total_time:.4f}초")
        
        
    def test_show(self,img):
        self.test_figure.clear()  # 이전 그래프 지우기
        ax = self.test_figure.add_subplot(111)
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        # canvas에 그리기
        self.test_canvas.figure = self.test_figure
        self.test_canvas.draw()
            

    

if __name__ == "__main__":
    app = QApplication(sys.argv)

    # WindowClass의 인스턴스 생성
    myWindow=WindowClass()
    # 프로그램 화면을 보여주는 코드
    myWindow.show()
    # 프로그램을 이벤트 루프로 진입시키는(프로그램을 작동시키는) 코드
    sys.exit(app.exec_())