# viewer.py

from  detect import detect
from plot import printPlot
from utils.general import strip_optimizer
import argparse

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import QApplication, QWidget
import torch
import time
import cv2
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QGridLayout, QPushButton, QStyle, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import os

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500,675)   # 첫 화면 크기
        MainWindow.move(0,100) # 첫 화면 위치
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        # 카메라 화면표시
        self.cam_frame = QtWidgets.QFrame(self.centralwidget)
        self.cam_frame.setGeometry(QtCore.QRect(15, 15, 1200, 675)) # 전체 프레임 안에서 (x , y , w, h)
        self.cam_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.cam_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.cam_frame.setObjectName("cam_frame")

        self.label_img_show = QtWidgets.QLabel(self.cam_frame)
        self.label_img_show.setGeometry(QtCore.QRect(10, 10, 1190, 670)) # 화면표시 프레임 안에서 (x , y , w, h)
        self.label_img_show.setObjectName("label_img_show")
        
        # 버튼 표시 프레임
        self.btn_frame = QtWidgets.QFrame(self.centralwidget)
        self.btn_frame.setGeometry(QtCore.QRect(1230,15,250,700)) # 버튼 프레임 안에서 (x , y , w, h)
        self.btn_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.btn_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.btn_frame.setObjectName("frame_3")

        # 버튼 레이아웃(세로)
        self.widget = QtWidgets.QWidget(self.btn_frame)
        self.widget.setGeometry(QtCore.QRect(0,0, 250, 700)) # 버튼 표시 프레임 안에서 (x , y , w, h)
        self.widget.setObjectName("widget")
        self.horizontalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.horizontalLayout.setContentsMargins(0, 10, 10, 0)
        self.horizontalLayout.setSpacing(10)
        self.horizontalLayout.setObjectName("horizontalLayout")
        
        # video source open
        self.btn_opencam = QtWidgets.QPushButton()
        self.btn_opencam.setObjectName("btn_opencam")
        self.horizontalLayout.addWidget(self.btn_opencam)

        self.btn_detect = QtWidgets.QPushButton(self.widget)
        self.btn_detect.setObjectName("btn_detect")
        self.horizontalLayout.addWidget(self.btn_detect)
        # 나가기
        self.btn_exit = QtWidgets.QPushButton(self.widget)
        self.btn_exit.setObjectName("btn_exit")
        self.horizontalLayout.addWidget(self.btn_exit)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1600, 17))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.w = None
        
        self.retranslateUi(MainWindow)
        self.btn_opencam.clicked.connect(self.opencam) # open video source
        self.btn_detect.clicked.connect(MainWindow.close) # 모델이 구현되면 기능 사용
        self.btn_exit.clicked.connect(MainWindow.close) # ui 종료
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "- A.eye -"))
        self.label_img_show.setText(_translate("MainWindow", ""))
        self.btn_opencam.setText(_translate("MainWindow", "학습 시작"))
        self.btn_detect.setText(_translate("MainWindow", "학습 종료"))
        self.btn_exit.setText(_translate("MainWindow", "나가기"))
        
    def opencam(self):
        vedio_file = 'run.mp4' # our saved video
        self.camcapture = cv2.VideoCapture(vedio_file)
        self.timer = QtCore.QTimer()
        self.timer.start()
        self.timer.setInterval(40) # 1s 주기
        self.timer.timeout.connect(self.camshow)

    def camshow(self):
        # global self.camimg
        self.ret, self.camimg = self.camcapture.read()
        if self.ret:
            camimg = cv2.cvtColor(self.camimg, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(camimg.data, camimg.shape[1], camimg.shape[0], QtGui.QImage.Format_RGB888)
            self.pixmap = QtGui.QPixmap(showImage)
            self.p = self.pixmap.scaled(1190, 670, QtCore.Qt.IgnoreAspectRatio)
            self.label_img_show.setPixmap(self.p)
    
class Another_Window(object):
    def setupUi(self,Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(1600,700)
        self.main_layout = QtWidgets.QVBoxLayout(Dialog)
        self.main_layout.setObjectName("centralwidget")
        
        self.frame_6 = QtWidgets.QFrame()
        self.frame_6.setFrameShape(QFrame.Panel | QFrame.Sunken)
        self.graph_frame = QtWidgets.QFrame()
        
        self.pixmap = QPixmap('fig1.png') # our saved image
        self.re_pixmap = self.pixmap.scaled(1200,675)
                
        self.label_graph_show = QtWidgets.QLabel(self.graph_frame)
        self.label_graph_show.setPixmap(self.re_pixmap)
        
        self.layout_6 = QGridLayout()

        study_time = []
        f = open("study_time.txt","r") # our saved text (times)
        while True:
            line = f.readline()
            if line == '' :
                break
            study_time.append(line)

        f.close()
        self.label1 = QLabel(study_time[0])
        self.label2 = QLabel(study_time[1])
        self.label3 = QLabel(study_time[2])

        self.label4 = QLabel('')
        self.label5 = QLabel('')
        self.label6 = QLabel('')
        self.btn_back = QPushButton('back')
        self.btn_back.setIcon(QIcon("back_icon.png"))
        
        self.label1.setFont(QFont('',15))
        self.label2.setFont(QFont('',15))
        self.label3.setFont(QFont('',15))
        
        self.layout_6.addWidget(self.label1)
        self.layout_6.addWidget(self.label2)
        self.layout_6.addWidget(self.label3)
        self.layout_6.addWidget(self.label4)
        self.layout_6.addWidget(self.label5)
        self.layout_6.addWidget(self.label6)
        self.layout_6.addWidget(self.btn_back)
        
        self.frame_6.setLayout(self.layout_6)
        
        self.spliter_9 = QSplitter(Qt.Horizontal)
        self.spliter_9.addWidget(self.frame_6)
        self.spliter_9.addWidget(self.graph_frame)
        
        self.spliter_9.setSizes([100,1000])
        
        self.main_layout.addWidget(self.spliter_9)

        self.main_layout.setGeometry(QtCore.QRect(300,200,1500,675))
        Dialog.move(0,100)

class parentWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.main_ui = Ui_MainWindow()
        self.main_ui.setupUi(self)
            
class childWindow(QDialog):
    def __init__(self):
        QDialog.__init__(self)
        self.child=Another_Window()
        self.child.setupUi(self)    

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='study02.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')           
    parser.add_argument('--save-path', type=str, default='run.mp4', help='file/dir/URL/glob')
    parser.add_argument('--mode', type=int, default=0, help='play mode') # 0 for user, 1 for developer
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()    
    with torch.no_grad():
        if args.update:  # update all models (to fix SourceChangeWarning)
            for args.weights in ['best.pt']:
                detect(args)
                strip_optimizer(args.weights)
        else:
            fps_cnt_list, total_fps_cnt_list, fps_obj_list = detect(args) 
            printPlot(fps_cnt_list, total_fps_cnt_list, fps_obj_list)
    
    app = QApplication(sys.argv)
    window=parentWindow()
    child=childWindow()
    btn=window.main_ui.btn_detect
    btn.clicked.connect(child.show)
    btn_back = child.child.btn_back
    btn_back.clicked.connect(window.show)
    window.show()
    sys.exit(app.exec_())
