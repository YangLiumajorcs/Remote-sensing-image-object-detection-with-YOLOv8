from PyQt5.Qt import *
import sys
from ultralytics import YOLO
from PIL import Image
from PIL import ImageQt
import time
import cv2
from window.window_ly import *
import threading




myevent = threading.Event()
class Yolov8_detect(threading.Thread):
    def __init__(self,Mainwindow = None):
        super(Yolov8_detect, self).__init__() #调用父类初始化函数
        self.model = YOLO("best.pt") #传递模型权重

        self.mode = None #检测模式
        self.continueLast = None #图片检测中用于保存上一次打开的图片文件路径
        self.iou = 0.5 #当前交并比
        self.conf = 0.5 #当前置信度
        self.latency = None #当前帧间延迟
        self.filename = None #打开的文件对象
        self.needdetect = False
        self.pictureResults =None #图片检测后的结果
        self.Mainwindow = Mainwindow
        #视频检测
        self.cap = None#打开视频文件
        self.frame = None #视频原始帧
        self.pause = False #暂停视频检测
    def setting(self,iou = None,conf = None,latency = None):
        self.iou = iou  # 设置交并比为交并比滑条
        self.conf = conf  # 设置置信度为置信度滑条
        self.latency = latency  # 设置帧间延迟为帧间延迟滑条
    def run(self):
        while True:
            if self.needdetect and self.mode == 0 and self.continueLast is None and self.filename:
                self.needdetect = False
                myevent.clear()
                self.continueLast = self.filename #缓存图片
                im1 = Image.open(self.filename)#读取图片转换为Qimage
                self.pictureResults = self.model.predict(source=im1,save=True,iou= self.iou,conf = self.conf)
                myevent.set()
            elif self.needdetect and self.mode == 0 and self.continueLast is not None:
                self.needdetect = False
                myevent.clear()
                im1 = Image.open(self.filename)
                self.pictureResults = self.model.predict(source=im1,save=True,iou=self.iou,conf = self.conf)
                myevent.set()
            elif self.needdetect and self.mode == 1 and self.filename:
                self.cap = cv2.VideoCapture(self.filename)
                while self.cap is not None and self.cap.isOpened():
                    if self.pause:
                        success,self.frame = self.cap.read()
                        if success:
                            self.pictureResults = self.model.predict(source = self.frame,save=True,iou = self.iou,conf = self.conf)

                            im_bgr = self.pictureResults[0].plot()
                            im_rgb = Image.fromarray(im_bgr[..., ::-1])

                            pixmap = ImageQt.toqpixmap(im_rgb)
                            self.Mainwindow.labelvideo.setPixmap(pixmap.scaled(self.Mainwindow.labelvideo.size(), aspectRatioMode = True))

                            aircraft = 0
                            oiltank = 0
                            overpass = 0
                            playground = 0
                            for item in self.pictureResults[0].boxes.cls:
                                if item == 0:
                                    aircraft += 1
                                elif item == 1:
                                    oiltank += 1
                                elif item == 2:
                                    overpass += 1
                                else:
                                    playground += 1
                            str1 = "检测结果:" + '\n'"imageSize:" + str(
                                self.pictureResults[0].orig_shape[0]) + ',' + str(
                                self.pictureResults[0].orig_shape[1]) + '\t'
                            str2 = "物体个数:" + str(aircraft) + ' aircraft  ' + str(oiltank) + ' oiltank ' + str(
                                overpass) + ' overpass ' + str(playground) \
                                   + ' playground '
                            str3 = "预处理速度" + str(
                                round(self.pictureResults[0].speed['preprocess'], 2)) + 'ms  ' + "处理速度" + str(
                                round(self.pictureResults[0].speed['inference'], 2)) + 'ms  ' + \
                                   "后处理速度" + str(round(self.pictureResults[0].speed['postprocess'], 2)) + 'ms'
                            mystr = str1 + str2 + str3
                            self.Mainwindow.mystr = self.Mainwindow.mystr+mystr + '\n'
                            self.Mainwindow.listWidget.addItem(mystr)
                            self.Mainwindow.listWidget.setCurrentRow(self.Mainwindow.listWidget.count() - 1)

                        else:
                            self.pause = False
                            self.Mainwindow.stopButton.setText("重新检测")
                            break


                       # self.Mainwindow.label_4.setPixmap(pixmap2.scaled(self.Mainwindow.label_4.size()))
            elif self.needdetect and self.mode == 2:
                self.cap = cv2.VideoCapture()
                flag = self.cap.open(1)
                if flag:
                    while self.pause:
                        success,frame = self.cap.read()
                        if success:
                            self.pictureResults = self.model.predict(source=frame,save=True,iou = self.iou,conf = self.conf)
                            im_bgr = self.pictureResults[0].plot()
                            im_rgb = Image.fromarray(im_bgr[..., ::-1])

                            pixmap = ImageQt.toqpixmap(im_rgb)
                            self.Mainwindow.labelvideo.setPixmap(
                                pixmap.scaled(self.Mainwindow.labelvideo.size(), aspectRatioMode=True))

                            aircraft = 0
                            oiltank = 0
                            overpass = 0
                            playground = 0
                            for item in self.pictureResults[0].boxes.cls:
                                if item == 0:
                                    aircraft += 1
                                elif item == 1:
                                    oiltank += 1
                                elif item == 2:
                                    overpass += 1
                                else:
                                    playground += 1
                            str1 = "检测结果:" + '\n'"imageSize:" + str(
                                self.pictureResults[0].orig_shape[0]) + ',' + str(
                                self.pictureResults[0].orig_shape[1]) + '\t'
                            str2 = "物体个数:" + str(aircraft) + ' aircraft  ' + str(oiltank) + ' oiltank ' + str(
                                overpass) + ' overpass ' + str(playground) \
                                   + ' playground '
                            str3 = "预处理速度" + str(
                                round(self.pictureResults[0].speed['preprocess'], 2)) + 'ms  ' + "处理速度" + str(
                                round(self.pictureResults[0].speed['inference'], 2)) + 'ms  ' + \
                                   "后处理速度" + str(round(self.pictureResults[0].speed['postprocess'], 2)) + 'ms'
                            mystr = str1 + str2 + str3
                            self.Mainwindow.mystr = self.Mainwindow.mystr + mystr + '\n'
                            self.Mainwindow.listWidget.addItem(mystr)
                            self.Mainwindow.listWidget.setCurrentRow(self.Mainwindow.listWidget.count() - 1)


class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)
        self.label_13.hide()
        self.label_15.hide()
        self.enable.hide()
        self.stopButton.hide()
        self.latency_slider.hide()
        self.label_14.hide()
        self.msg_box = QMessageBox(self)
        self.mystr = ''
        self.yolo = Yolov8_detect(self)
        self.yolo.start()
    def picture_detection(self):#点击导入图片按钮触发事件
        self.stopButton.hide()
        self.pushButton.setGeometry(QtCore.QRect(1130, 630, 81, 61))
        self.yolo.needdetect = False
        self.label_4.clear()
        self.label_6.clear()
        if self.yolo.cap is not None:
            self.yolo.cap.release()
            self.yolo.cap = None
        self.groupBox.setVisible(True)
        self.groupBox_2.setVisible(True)
        self.groupBoxvideo.setVisible(False)

        self.yolo.filename = None

        self.yolo.continueLast = None #清空缓存图片
        #创建一个线程对象用于后期显示图片
        self.yolo.mode = 0  # 设置检测模式

        #打开文件框读取文件路径
        self.yolo.filename ,_ = QFileDialog.getOpenFileName(self, "选择图片",
                                                  "/Users/liuyang/Documents/biyelunwen/ROSDdataset/images/test",
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)")
        if self.yolo.filename:
            t1 = threading.Thread(target=self.show_picture_detect)
            t1.start()
        self.yolo.needdetect = True#需要进行一次检测

        #self.show_picture_detect()

    def show_picture_detect(self):
        # 显示待检测区
        if self.yolo.mode == 0:
            myevent.wait()
            pixmap = QPixmap(self.yolo.filename).scaled(self.label_4.size())
            #self.label_4.setPixmap(pixmap)
            # 显示输出区
            im_bgr = self.yolo.pictureResults[0].plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            pixmap1 = ImageQt.toqpixmap(im_rgb)
            self.label_6.setPixmap(pixmap1.scaled(self.label_6.size()))
            self.label_4.setPixmap(pixmap)
            aircraft = 0
            oiltank = 0
            overpass = 0
            playground = 0
            for item in self.yolo.pictureResults[0].boxes.cls:
                if item == 0:
                    aircraft += 1
                elif item == 1:
                    oiltank += 1
                elif item == 2:
                    overpass += 1
                else:
                    playground += 1
            str1 = "检测结果:" + '\n'"imageSize:" + str(self.yolo.pictureResults[0].orig_shape[0]) + ',' + str(\
                self.yolo.pictureResults[0].orig_shape[1]) + '\t'
            str2 = "物体个数:" + str(aircraft) + ' aircraft  ' + str(oiltank) + ' oiltank ' + str(\
                overpass) + ' overpass ' + str(playground) \
                   + ' playground '
            str3 = "预处理速度" + str(\
                round(self.yolo.pictureResults[0].speed['preprocess'], 2)) + 'ms  ' + "处理速度" + str(
                round(self.yolo.pictureResults[0].speed['inference'], 2)) + 'ms  ' + \
                   "后处理速度" + str(round(self.yolo.pictureResults[0].speed['postprocess'], 2)) + 'ms'
            mystr = str1 + str2 +str3
            self.mystr = self.mystr + mystr + '\n'
            self.listWidget.addItem(mystr)
            print(1)
            self.listWidget.setCurrentRow(self.listWidget.count() - 1)
            myevent.clear()



    def video_detection(self):
        self.stopButton.show()
        self.stopButton.setGeometry(QtCore.QRect(1080, 630, 81, 61))
        self.pushButton.setGeometry(QtCore.QRect(1170, 630, 81, 61))
        self.yolo.needdetect = False
        self.stopButton.setText("开始检测")
        self.labelvideo.clear()
        self.yolo.filename = None
        if self.yolo.cap is not None:
            self.yolo.cap.release()
            self.yolo.cap = None
        self.groupBox.setVisible(False)
        self.groupBox_2.setVisible(False)
        self.groupBoxvideo.setVisible(True)
        self.labelvideo.clear()


        self.yolo.mode = 1 #模式设置为视频检测

        self.yolo.filename, _ = QFileDialog.getOpenFileName(self, "选择图片", "/Users/liuyang",
                                                    "video files (*.mov *.mp4)")
        self.yolo.needdetect = True

    def camera_detection(self):
        self.yolo.mode = 2
        if self.yolo.cap is not None:
            self.yolo.cap.release()
            self.yolo.cap = None

        self.stopButton.show()
        self.stopButton.setGeometry(QtCore.QRect(1080, 630, 81, 61))
        self.pushButton.setGeometry(QtCore.QRect(1170, 630, 81, 61))
        self.stopButton.setText("开始检测")
        self.groupBox.setVisible(False)
        self.groupBox_2.setVisible(False)
        self.groupBoxvideo.setVisible(True)
        self.yolo.needdetect = True

        self.yolo.continueLast = None
        self.yolo.filename = None
        self.yolo.pause = False
        time.sleep(0.5)
        self.labelvideo.setPixmap(QPixmap(""))

    def saveResults(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save File", "", "Text Files (*.txt)")

        if file_path:
            # 将字符串内容写入文件
            with open(file_path, 'w') as file:
                file.write(self.mystr)

            print(f"File saved to: {file_path}")
    #改变配置信息
    def anchor_sliderfun(self):
        self.yolo.setting(iou=float(self.anchor_slider.value() / 100),
                                conf=float(self.confidence_slider.value() / 100),latency=self.latency_slider.value())
        if self.yolo.mode == 0:
            t1 = threading.Thread(target=self.show_picture_detect)
            t1.start()
        self.yolo.needdetect = True
    def confidence_sliderfun(self):
        self.yolo.setting(iou=float(self.anchor_slider.value() / 100),
                                conf=float(self.confidence_slider.value() / 100), latency=self.latency_slider.value())
        if self.yolo.mode == 0:
            t1 = threading.Thread(target=self.show_picture_detect)
            t1.start()
        self.yolo.needdetect = True
    def latency_sliderfun(self):
        self.yolo.setting(iou=float(self.anchor_slider.value() / 100),
                                conf=float(self.confidence_slider.value() / 100), latency=self.latency_slider.value())
        if self.yolo.mode == 0:
            t1 = threading.Thread(target=self.show_picture_detect)
            t1.start()
        self.yolo.needdetect = True

    #改变标签信息
    def iouFun(self):  # 实时更改iou标签值
        self.label_10.setText(f"{str(self.anchor_slider.value() / 100)}")
    def conFun(self):#实时更改置信度标签值
        self.label_12.setText(f"{str(self.confidence_slider.value()/100)}")
    def latencyFun(self):#实时更改帧间延时标签值
        self.label_14.setText(f"{self.latency_slider.value()}")

    def combchanged(self):
        if self.comb.currentIndex() == 0:
            self.yolo.model = YOLO("weights/YOLOv8/weights/best.pt")
        elif self.comb.currentIndex() == 1:
            self.yolo.model = YOLO("weights/YOLO_MHSA/weights/best.pt")
        elif self.comb.currentIndex() == 2:
            self.yolo.model = YOLO("weights/YOLOv8_LSK/weights/best.pt")
        elif self.comb.currentIndex() == 3:
            self.yolo.model = YOLO("weights/YOLOv8_MHSA_LSK/weights/best.pt")

    #停止视频检测
    def stopvideo(self):
        if self.yolo.mode == 1 and self.yolo.cap.isOpened() and self.yolo.pause:#如果当前检测模式是视频并且cap打开
            self.yolo.pause = False
            self.stopButton.setText("继续检测")
        elif self.yolo.mode == 1 and self.yolo.cap.isOpened() and self.yolo.pause == False:
            self.yolo.pause = True
            self.stopButton.setText("停止检测")
        elif self.yolo.mode == 2 and self.yolo.pause:
            self.yolo.pause = False
            self.stopButton.setText("继续检测")
        elif self.yolo.mode == 2 and self.yolo.pause == False:
            self.yolo.pause = True
            self.stopButton.setText("停止检测")
app = QApplication(sys.argv)
myWin = MainWindow()
myWin.show()
sys.exit(app.exec_())
