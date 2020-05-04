# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui_ocr.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidget, QFrame, QTableWidgetItem, QHeaderView
import pymysql


class Ui_ui_ocr(object):
    def setupUi(self, ui_ocr):
        ui_ocr.setObjectName("ui_ocr")
        ui_ocr.resize(1300, 800)
        self.ocr_ui = QtWidgets.QWidget(ui_ocr)
        self.ocr_ui.setObjectName("ocr_ui")
        self.line = QtWidgets.QFrame(self.ocr_ui)
        self.line.setGeometry(QtCore.QRect(230, -10, 20, 800))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self.ocr_ui)
        self.line_2.setGeometry(QtCore.QRect(40, 130, 151, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.ocr = QtWidgets.QPushButton(self.ocr_ui)
        self.ocr.setGeometry(QtCore.QRect(10, 170, 211, 101))
        self.ocr.setIconSize(QtCore.QSize(16, 16))
        self.ocr.setObjectName("ocr")
        self.his = QtWidgets.QPushButton(self.ocr_ui)
        self.his.setGeometry(QtCore.QRect(10, 300, 211, 101))
        self.his.setObjectName("his")
        self.logo = QtWidgets.QLabel(self.ocr_ui)
        self.logo.setGeometry(QtCore.QRect(40, 20, 151, 101))
        self.logo.setText("")
        self.logo.setObjectName("logo")
        self.stackedWidget = QtWidgets.QStackedWidget(self.ocr_ui)
        self.stackedWidget.setGeometry(QtCore.QRect(260, 20, 1011, 721))
        self.stackedWidget.setObjectName("stackedWidget")

        # 这里添加了堆叠布局的第一页
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.welcome = QtWidgets.QLabel(self.page)
        self.welcome.setGeometry(QtCore.QRect(160, 110, 671, 481))
        font = QtGui.QFont()
        font.setPointSize(30)
        self.welcome.setFont(font)
        self.welcome.setAlignment(QtCore.Qt.AlignCenter)
        self.welcome.setObjectName("welcome")
        self.stackedWidget.addWidget(self.page)
        # 这里结束添加了堆叠布局的第一页

        # 这里开始添加了堆叠布局的第二页
        self.ocr_page = QtWidgets.QWidget()
        self.ocr_page.setObjectName("ocr_page")
        self.layoutWidget = QtWidgets.QWidget(self.ocr_page)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 30, 991, 561))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setHorizontalSpacing(10)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.img_show = QtWidgets.QLabel(self.layoutWidget)
        self.img_show.setMaximumSize(QtCore.QSize(653, 559))
        self.img_show.setFrameShape(QtWidgets.QFrame.Box)
        self.img_show.setText("")
        self.img_show.setObjectName("img_show")
        self.gridLayout_3.addWidget(self.img_show, 0, 0, 2, 1)
        self.recoresult = QtWidgets.QLabel(self.layoutWidget)
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(30)
        self.recoresult.setFont(font)
        self.recoresult.setFrameShape(QtWidgets.QFrame.Box)
        self.recoresult.setObjectName("recoresult")
        self.gridLayout_3.addWidget(self.recoresult, 0, 1, 1, 1)
        self.result_ = QtWidgets.QLabel(self.layoutWidget)
        self.result_.setFrameShape(QtWidgets.QFrame.Box)
        self.result_.setText("")
        self.result_.setObjectName("result_")
        self.gridLayout_3.addWidget(self.result_, 1, 1, 1, 1)
        self.gridLayout_3.setColumnStretch(0, 2)
        self.gridLayout_3.setColumnStretch(1, 1)
        self.gridLayout_3.setRowStretch(0, 1)
        self.gridLayout_3.setRowStretch(1, 6)
        self.reco = QtWidgets.QPushButton(self.ocr_page)
        self.reco.setGeometry(QtCore.QRect(680, 620, 191, 71))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(25)
        self.reco.setFont(font)
        self.reco.setObjectName("reco")
        self.clean = QtWidgets.QPushButton(self.ocr_page)
        self.clean.setGeometry(QtCore.QRect(450, 620, 211, 71))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(25)
        self.clean.setFont(font)
        self.clean.setObjectName("clean")
        self.open = QtWidgets.QPushButton(self.ocr_page)
        self.open.setGeometry(QtCore.QRect(10, 620, 211, 71))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(25)
        self.open.setFont(font)
        self.open.setObjectName("open")
        self.copyget = QtWidgets.QPushButton(self.ocr_page)
        self.copyget.setGeometry(QtCore.QRect(880, 620, 121, 71))
        font = QtGui.QFont()
        font.setFamily("宋体")
        font.setPointSize(20)
        self.copyget.setFont(font)
        self.copyget.setObjectName("copyget")
        self.stackedWidget.addWidget(self.ocr_page)
        # 这里结束添加了堆叠布局的第二页

        # 这里开始添加了堆叠布局的第三页
        self.his_page = QtWidgets.QWidget()
        self.instructions = QtWidgets.QLabel(self.his_page)
        self.instructions.setGeometry(QtCore.QRect(230, 30, 521, 91))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.instructions.setFont(font)
        self.instructions.setAlignment(QtCore.Qt.AlignCenter)
        self.instructions.setObjectName("instructions")

        db = pymysql.connect("localhost", "root", "123456", "ocrhis", charset='utf8')
        # 获取游标、数据
        cur = db.cursor()
        cur.execute("SELECT * FROM his")
        data = cur.fetchall()

        # 数据列名
        col_lst = ['序号', '文件路径', '识别时间', '识别结果']

        # 数据的大小
        row = len(data)
        vol = len(data[0])

        # 插入表格
        self.histable = QTableWidget(row, vol, self.his_page)
        self.histable.setGeometry(QtCore.QRect(100, 130, 821, 521))
        font = QtGui.QFont('微软雅黑', 10)

        # 设置字体、表头
        self.histable.horizontalHeader().setFont(font)
        self.histable.setHorizontalHeaderLabels(col_lst)
        # 设置竖直方向表头不可见
        self.histable.verticalHeader().setVisible(False)
        self.histable.setFrameShape(QFrame.Box)

        # 构建表格插入数据
        for i in range(row):
            for j in range(vol):
                temp_data = data[i][j]  # 临时记录，不能直接插入表格
                data1 = QTableWidgetItem(str(temp_data))  # 转换后可插入表格
                self.histable.setItem(i, j, data1)
        self.histable.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.histable.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.histable.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.histable.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.his_page.setObjectName("his_page")
        self.stackedWidget.addWidget(self.his_page)
        # 这里结束添加了堆叠布局的第三页

        self.retranslateUi(ui_ocr)
        self.stackedWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(ui_ocr)

    def retranslateUi(self, ui_ocr):
        _translate = QtCore.QCoreApplication.translate
        ui_ocr.setWindowTitle(_translate("ui_ocr", "MainWindow"))
        self.ocr.setText(_translate("ui_ocr", "文字识别"))
        self.his.setText(_translate("ui_ocr", "识别历史"))
        self.welcome.setText(_translate("ui_ocr", "欢迎使用OCR系统"))
        self.instructions.setText(_translate("ui_ocr", "简单ocr识别记录，双击可复制单元格其内容"))
        self.recoresult.setText(_translate("ui_ocr", "识别结果："))
        self.reco.setText(_translate("ui_ocr", "开始识别"))
        self.clean.setText(_translate("ui_ocr", "清除图片"))
        self.open.setText(_translate("ui_ocr", "上传图片"))
        self.copyget.setText(_translate("ui_ocr", "复制结果"))
