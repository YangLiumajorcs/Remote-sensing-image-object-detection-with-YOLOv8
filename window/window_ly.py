# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'YOLOv8.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from imgs import ahu

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(1261, 713)
        MainWindow.setMinimumSize(QtCore.QSize(1261, 713))
        MainWindow.setMaximumSize(QtCore.QSize(1261, 713))
        font = QtGui.QFont()
        font.setPointSize(13)
        font.setBold(False)
        font.setItalic(False)
        font.setWeight(50)
        MainWindow.setFont(font)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("ahu.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setWindowOpacity(1.0)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setAnimated(True)
        MainWindow.setDocumentMode(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        MainWindow.setDockNestingEnabled(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setTabletTracking(False)
        self.centralwidget.setAcceptDrops(False)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox1 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox1.setEnabled(True)
        self.groupBox1.setGeometry(QtCore.QRect(0, 0, 1261, 71))
        self.groupBox1.setTitle("")
        self.groupBox1.setObjectName("groupBox1")
        self.label = QtWidgets.QLabel(self.groupBox1)
        self.label.setGeometry(QtCore.QRect(0, 0, 71, 71))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap(":/新前缀/ahu.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.groupBox1)
        self.label_2.setGeometry(QtCore.QRect(70, 0, 241, 71))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(False)
        font.setItalic(True)
        font.setUnderline(False)
        font.setWeight(50)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(0, 70, 531, 541))
        self.groupBox.setTitle("")
        self.groupBox.setObjectName("groupBox")
        self.label_3 = QtWidgets.QLabel(self.groupBox)
        self.label_3.setGeometry(QtCore.QRect(0, 0, 531, 51))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(False)
        font.setStrikeOut(False)
        self.label_3.setFont(font)
        self.label_3.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")

        self.label_4 = QtWidgets.QLabel(self.groupBox)
        self.label_4.setGeometry(QtCore.QRect(0, 50, 531, 491))
        self.label_4.setText("")
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")

        self.groupBoxvideo = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBoxvideo.setGeometry(QtCore.QRect(0,70,1062,541))

        self.labelvideo = QtWidgets.QLabel(self.groupBoxvideo)
        self.labelvideo.setGeometry(QtCore.QRect(0,50,1062,491))
        self.labelvideo.setAlignment(QtCore.Qt.AlignCenter)
        self.labelvideo.setStyleSheet("background-color:black")
        self.labelvideo.setFont(font)
        #self.labelvideo.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")



        self.groupBoxvideo.setVisible(False)

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(530, 70, 531, 541))
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.label_5 = QtWidgets.QLabel(self.groupBox_2)
        self.label_5.setGeometry(QtCore.QRect(0, 0, 531, 51))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(False)
        self.label_5.setFont(font)
        self.label_5.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.groupBox_2)
        self.label_6.setGeometry(QtCore.QRect(0, 50, 531, 481))
        self.label_6.setStyleSheet("x")
        self.label_6.setText("")
        self.label_6.setObjectName("label_6")
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(1060, 70, 201, 541))
        self.groupBox_3.setTitle("")
        self.groupBox_3.setObjectName("groupBox_3")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(0, 0, 201, 51))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setItalic(False)
        self.label_7.setFont(font)
        self.label_7.setStyleSheet("background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 rgba(0, 0, 0, 255), stop:1 rgba(255, 255, 255, 255));")
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(0, 50, 201, 51))
        self.label_8.setObjectName("label_8")
        self.picture = QtWidgets.QPushButton(self.groupBox_3)
        self.picture.setGeometry(QtCore.QRect(0, 100, 61, 51))
        self.picture.setText("")
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/新前缀/picture.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.picture.setIcon(icon1)
        self.picture.setObjectName("picture")
        self.video = QtWidgets.QPushButton(self.groupBox_3)
        self.video.setGeometry(QtCore.QRect(70, 100, 61, 51))
        self.video.setText("")
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/新前缀/video.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.video.setIcon(icon2)
        self.video.setObjectName("video")
        self.camera = QtWidgets.QPushButton(self.groupBox_3)
        self.camera.setGeometry(QtCore.QRect(140, 100, 61, 51))
        self.camera.setText("")
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/新前缀/camera.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.camera.setIcon(icon3)
        self.camera.setObjectName("camera")
        self.label_9 = QtWidgets.QLabel(self.groupBox_3)
        self.label_9.setGeometry(QtCore.QRect(0, 160, 201, 41))
        self.label_9.setObjectName("label_9")
        self.label_10 = QtWidgets.QLabel(self.groupBox_3)
        self.label_10.setGeometry(QtCore.QRect(0, 210, 31, 31))
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.anchor_slider = QtWidgets.QSlider(self.groupBox_3)
        self.anchor_slider.setGeometry(QtCore.QRect(30, 210, 161, 31))
        self.anchor_slider.setMaximum(100)
        self.anchor_slider.setSingleStep(1)
        self.anchor_slider.setPageStep(0)
        self.anchor_slider.setProperty("value", 50)
        self.anchor_slider.setOrientation(QtCore.Qt.Horizontal)
        self.anchor_slider.setObjectName("anchor_slider")

        self.label_11 = QtWidgets.QLabel(self.groupBox_3)
        self.label_11.setGeometry(QtCore.QRect(0, 250, 201, 31))
        self.label_11.setObjectName("label_11")

        self.label_quan = QtWidgets.QLabel(self.groupBox_3)
        self.label_quan.setGeometry(QtCore.QRect(0, 330, 201, 31))
        self.label_quan.setObjectName("label_quan")

        self.comb = QtWidgets.QComboBox(self.groupBox_3)
        self.comb.addItem("YOLOv8")
        self.comb.addItem("YOLOv8_MHSA")
        self.comb.addItem("YOLOv8_LSK")
        self.comb.addItem("YOLOv8_MHSA_LSK")
        self.comb.setGeometry(QtCore.QRect(0,360,201,31))


        self.label_12 = QtWidgets.QLabel(self.groupBox_3)
        self.label_12.setGeometry(QtCore.QRect(0, 280, 31, 41))
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.confidence_slider = QtWidgets.QSlider(self.groupBox_3)
        self.confidence_slider.setGeometry(QtCore.QRect(30, 290, 160, 22))
        self.confidence_slider.setMaximum(100)
        self.confidence_slider.setProperty("value", 50)
        self.confidence_slider.setOrientation(QtCore.Qt.Horizontal)
        self.confidence_slider.setObjectName("confidence_slider")
        self.label_13 = QtWidgets.QLabel(self.groupBox_3)
        self.label_13.setGeometry(QtCore.QRect(0, 330, 191, 31))
        self.label_13.setObjectName("label_13")
        self.latency_slider = QtWidgets.QSlider(self.groupBox_3)
        self.latency_slider.setGeometry(QtCore.QRect(29, 390, 171, 22))
        self.latency_slider.setMaximum(24)
        self.latency_slider.setMinimum(1)
        self.latency_slider.setProperty("value", 10)
        self.latency_slider.setOrientation(QtCore.Qt.Horizontal)
        self.latency_slider.setObjectName("latency_slider")
        self.label_14 = QtWidgets.QLabel(self.groupBox_3)
        self.label_14.setGeometry(QtCore.QRect(0, 380, 31, 41))
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.enable = QtWidgets.QRadioButton(self.groupBox_3)
        self.enable.setGeometry(QtCore.QRect(139, 330, 51, 31))
        self.enable.setObjectName("enable")
        self.label_15 = QtWidgets.QLabel(self.groupBox_3)
        self.label_15.setGeometry(QtCore.QRect(80, 330, 41, 31))
        self.label_15.setObjectName("label_15")

        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        self.listWidget.setGeometry(QtCore.QRect(0, 610, 1061, 100))
        self.listWidget.setObjectName("listWidget")

        self.stopButton = QtWidgets.QPushButton(self.centralwidget)
        self.stopButton.setGeometry(QtCore.QRect(1050,630,81,61))



        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(1130, 630, 81, 61))
        self.pushButton.setObjectName("pushButton")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.picture.clicked.connect(MainWindow.picture_detection) # type: ignore
        self.video.clicked.connect(MainWindow.video_detection) # type: ignore
        self.camera.clicked.connect(MainWindow.camera_detection) # type: ignore
        self.pushButton.clicked.connect(MainWindow.saveResults) # type: ignore
        self.latency_slider.valueChanged['int'].connect(MainWindow.latencyFun) # type: ignore
        self.confidence_slider.valueChanged['int'].connect(MainWindow.conFun) # type: ignore
        self.anchor_slider.valueChanged['int'].connect(MainWindow.iouFun) # type: ignore
        self.anchor_slider.sliderReleased.connect( MainWindow.anchor_sliderfun)
        self.confidence_slider.sliderReleased.connect(MainWindow.confidence_sliderfun)
        self.latency_slider.sliderReleased.connect(MainWindow.latency_sliderfun)
        self.stopButton.clicked.connect(MainWindow.stopvideo)
        self.comb.currentIndexChanged.connect(MainWindow.combchanged)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "YOLOv8遥感目标检测系统"))
        self.label_2.setText(_translate("MainWindow", "YOLOv8遥感图像目标检测系统"))
        self.label_3.setText(_translate("MainWindow", "检测区"))
        self.label_5.setText(_translate("MainWindow", "输出区"))
        self.label_7.setText(_translate("MainWindow", "设置"))
        self.label_8.setText(_translate("MainWindow", "检测模式"))
        self.label_9.setText(_translate("MainWindow", "锚框最大交并比"))
        self.label_10.setText(_translate("MainWindow", "0"))
        self.label_11.setText(_translate("MainWindow", "置信度"))
        self.label_quan.setText(_translate("MainWindow","权重选择"))
        self.label_12.setText(_translate("MainWindow", "0"))
        self.label_13.setText(_translate("MainWindow", "帧间延迟"))
        self.label_14.setText(_translate("MainWindow", "0"))
        self.enable.setText(_translate("MainWindow", "启用"))
        self.label_15.setText(_translate("MainWindow", "fps:0"))
        self.label_10.setText(_translate("MainWindow","0.5"))
        self.label_12.setText(_translate("MainWindow", "0.5"))
        self.label_14.setText(_translate("MainWindow", "10"))
        self.pushButton.setText(_translate("MainWindow", "保存结果"))
        self.labelvideo.setText(_translate("MainWindow","视频检测区"))
        self.stopButton.setText(_translate("MainWindow","停止检测"))

